"""Command-line entrypoint (`pfas-bench`) for modular RPA1163 PFAS benchmark workflows."""

import argparse
import json
from pathlib import Path
from typing import List

from rdkit import Chem

from pfas_bench.config import BenchmarkConfig
from pfas_bench.place_substrate import place_substrate_in_pocket
from pfas_bench.prep_ligands import audit_fluorocarbon_torsion, prepare_anionic_substrate
from pfas_bench.prep_receptor import (
    build_receptor_variant,
    evaluate_fold_sanity_gate,
    extract_3r3v_reference,
)
from pfas_bench.run_md import run_holo_replicate
from pfas_bench.score_and_plot import rank_variants_table, render_ranking_plot


def cmd_prep_ligands(args: argparse.Namespace) -> None:
    cfg = BenchmarkConfig.from_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = out_dir / "prepared_substrates.sdf"
    meta_rows = []
    with Chem.SDWriter(str(sdf_path)) as writer:
        for sid, spec in cfg.substrates.items():
            prep = prepare_anionic_substrate(spec, charge_method_priority=cfg.charge_method_priority)
            prep.rd_mol.SetProp("charge_method", prep.charge_method)
            prep.rd_mol.SetProp("net_charge_e", str(prep.net_charge_e))
            writer.write(prep.rd_mol)
            meta_rows.append({
                "substrate_id": sid,
                "smiles": spec.smiles,
                "charge_method": prep.charge_method,
                "net_charge_e": prep.net_charge_e,
            })
            print(f"[prep-ligands] {sid}: charge_method={prep.charge_method}, net_q={prep.net_charge_e:+.3f} e")

    with open(out_dir / "prepared_substrates.json", "w", encoding="utf-8") as fh:
        json.dump(meta_rows, fh, indent=2)
    print(f"[prep-ligands] Saved 3D extended substrates to {sdf_path}")


def cmd_torsion_audit(args: argparse.Namespace) -> None:
    cfg = BenchmarkConfig.from_yaml(args.config)
    spec = cfg.substrates[args.substrate]
    prep = prepare_anionic_substrate(spec, charge_method_priority=cfg.charge_method_priority)
    c_idxs = [a.GetIdx() for a in prep.rd_mol.GetAtoms() if a.GetSymbol() == "C"]
    if len(c_idxs) < 4:
        raise ValueError(f"Substrate {args.substrate} has fewer than 4 carbons ({len(c_idxs)}); no C-C-C-C rotor.")
    dihed = (c_idxs[0], c_idxs[1], c_idxs[2], c_idxs[3])
    df, summary = audit_fluorocarbon_torsion(prep, dihed, step_deg=args.step_deg)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[torsion-audit] {args.substrate} dihedral {dihed}: {summary} -> {out_csv}")


def cmd_run_complex(args: argparse.Namespace) -> None:
    cfg = BenchmarkConfig.from_yaml(args.config)
    vspec = cfg.variants[args.variant]
    sspec = cfg.substrates[args.substrate]

    ref = extract_3r3v_reference(cfg.cif_holo_path, zone0_seqs=cfg.zone0_residues)
    prep_lig = prepare_anionic_substrate(sspec, charge_method_priority=cfg.charge_method_priority)
    placed = place_substrate_in_pocket(prep_lig.rd_mol, ref, torsion_step_deg=cfg.torsion_sweep_step_deg)

    out_dir = Path(args.output_dir) / f"{vspec.id}__{sspec.id}"
    rec_pdb = out_dir / f"receptor_{vspec.id}.pdb"
    prep_rec = build_receptor_variant(
        cfg.cif_holo_path,
        ref=ref,
        variant=vspec,
        assembly=args.assembly,
        ph=args.ph,
        out_pdb=rec_pdb,
    )
    fold_gate = evaluate_fold_sanity_gate(
        rec_pdb, ref=ref, rubric=cfg.rubric, is_experimental_or_homology=True
    )

    outer_k = 0.0 if args.assembly == "asymmetric_homodimer_AB" else args.outer_restraint_k
    rep_rows: List[dict] = []
    for rep in range(1, args.replicates + 1):
        print(
            f"[run-complex] {vspec.id} + {sspec.id} | rep {rep}/{args.replicates} | "
            f"solvent={args.solvent} | T={args.temperature_k}K | pH={args.ph}"
        )
        summary = run_holo_replicate(
            receptor=prep_rec,
            ligand=prep_lig,
            placed=placed,
            ref=ref,
            rubric=cfg.rubric,
            out_dir=out_dir,
            replicate_id=rep,
            solvent_mode=args.solvent,
            temperature_K=args.temperature_k,
            divalent_CaCl2_mM=args.cacl2_mm,
            outer_restraint_kcal_mol_A2=outer_k,
            equilibration_steps=args.equil_steps,
            production_steps=args.prod_steps,
            report_interval=args.report_interval,
        )
        rep_rows.append(summary)
        print(
            f"   -> P_NAC={summary['p_nac_pct']:.1f}% | E_pull={summary['mean_E_pull_MV_cm']:+.1f} MV/cm | "
            f"d_att={summary['mean_d_attack_A']:.2f}A | ang={summary['mean_ang_SN2_deg']:.1f}deg | "
            f"step2_wat={summary['f_wat_step2']:.2f} | inv={summary['f_inverted_tail']:.2f}"
        )

    payload = {"fold_gate": fold_gate, "replicates": rep_rows}
    out_json = out_dir / "replicate_metrics.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[run-complex] Saved replicate metrics to {out_json}")


def cmd_run_panel(args: argparse.Namespace) -> None:
    """Runs a multi-variant x multi-substrate explicit-solvent benchmark panel and generates S_total ranking."""
    cfg = BenchmarkConfig.from_yaml(args.config)
    variant_ids = args.variants if args.variants else list(cfg.variants.keys())
    substrate_ids = args.substrates if args.substrates else list(cfg.substrates.keys())

    ref = extract_3r3v_reference(cfg.cif_holo_path, zone0_seqs=cfg.zone0_residues)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare & place each substrate once
    prep_ligs = {}
    placed_ligs = {}
    for sid in substrate_ids:
        sspec = cfg.substrates[sid]
        pl = prepare_anionic_substrate(sspec, charge_method_priority=cfg.charge_method_priority)
        prep_ligs[sid] = pl
        placed_ligs[sid] = place_substrate_in_pocket(pl.rd_mol, ref, torsion_step_deg=cfg.torsion_sweep_step_deg)
        print(
            f"[panel-prep] {sid}: charge={pl.charge_method} ({pl.net_charge_e:+.3f} e) | "
            f"torsion={placed_ligs[sid].selected_torsion_deg} | min_Z0_clear={placed_ligs[sid].min_zone0_clearance_A:.2f}A",
            flush=True,
        )

    all_reps: List[dict] = []
    fold_gates = {}
    outer_k = 0.0 if args.assembly == "asymmetric_homodimer_AB" else args.outer_restraint_k

    for vid in variant_ids:
        vspec = cfg.variants[vid]
        rec_pdb = out_root / "receptors" / f"receptor_{vspec.id}.pdb"
        prep_rec = build_receptor_variant(
            cfg.cif_holo_path, ref=ref, variant=vspec, assembly=args.assembly, ph=args.ph, out_pdb=rec_pdb
        )
        fg = evaluate_fold_sanity_gate(rec_pdb, ref=ref, rubric=cfg.rubric, is_experimental_or_homology=True)
        fold_gates[vid] = fg
        print(
            f"[panel-receptor] {vid}: Zone0_RMSD={fg['zone0_heavy_rmsd_A']:.3f}A | fold_pass={fg['stage_a_fold_pass']}",
            flush=True,
        )

        for sid in substrate_ids:
            pair_dir = out_root / "runs" / f"{vid}__{sid}"
            cached_json = pair_dir / "replicate_metrics.json"
            if getattr(args, "resume", False) and cached_json.exists():
                with open(cached_json, "r", encoding="utf-8") as fh:
                    cdata = json.load(fh)
                creps = cdata.get("replicates", [])
                if len(creps) >= args.replicates:
                    print(f"[panel-cache] Reusing completed {vid} + {sid} ({len(creps)} reps)", flush=True)
                    all_reps.extend(creps)
                    continue
            pair_reps = []
            for rep in range(1, args.replicates + 1):
                print(
                    f"[panel-md] {vid} + {sid} (rep {rep}/{args.replicates}, {args.solvent})...",
                    flush=True,
                )
                summary = run_holo_replicate(
                    receptor=prep_rec,
                    ligand=prep_ligs[sid],
                    placed=placed_ligs[sid],
                    ref=ref,
                    rubric=cfg.rubric,
                    out_dir=pair_dir,
                    replicate_id=rep,
                    solvent_mode=args.solvent,
                    temperature_K=args.temperature_k,
                    divalent_CaCl2_mM=args.cacl2_mm,
                    outer_restraint_kcal_mol_A2=outer_k,
                    equilibration_steps=args.equil_steps,
                    production_steps=args.prod_steps,
                    report_interval=args.report_interval,
                    resume=bool(getattr(args, "resume", False)),
                )
                pair_reps.append(summary)
                all_reps.append(summary)
                print(
                    f"   -> P_NAC={summary['p_nac_pct']:.1f}% (near={summary['p_near_nac_pct']:.1f}%) | "
                    f"E_pull={summary['mean_E_pull_MV_cm']:+.1f} MV/cm | d_att={summary['mean_d_attack_A']:.2f}A "
                    f"(min {summary['min_d_attack_A']:.2f}A) | ang={summary['mean_ang_SN2_deg']:.1f}deg "
                    f"(max {summary['max_ang_SN2_deg']:.1f}deg) | HB={summary['mean_cradle_hbonds']:.2f} | "
                    f"wat2={summary['f_wat_step2']:.2f} | inv={summary['f_inverted_tail']:.2f}",
                    flush=True,
                )
            with open(pair_dir / "replicate_metrics.json", "w", encoding="utf-8") as fh:
                json.dump({"fold_gate": fg, "replicates": pair_reps}, fh, indent=2)

    df_ranked = rank_variants_table(all_reps, cfg.rubric, fold_gates_by_variant=fold_gates)
    prefix = getattr(args, "summary_prefix", "summary_ranking")
    out_csv = out_root / f"{prefix}.csv"
    out_png = out_root / f"{prefix}.png"
    df_ranked.to_csv(out_csv, index=False)
    render_ranking_plot(df_ranked, out_png)
    print(f"\n=== EXPLICIT-SOLVENT LEADERBOARD ({out_csv}) ===", flush=True)
    print(df_ranked.to_string(index=False), flush=True)
    print(f"[panel-done] Plot written to {out_png}", flush=True)


def cmd_rank(args: argparse.Namespace) -> None:
    cfg = BenchmarkConfig.from_yaml(args.config)
    in_dir = Path(args.results_dir)
    all_reps = []
    fold_gates = {}
    for jpath in sorted(in_dir.glob("*/replicate_metrics.json")):
        with open(jpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        reps = data.get("replicates", [])
        if reps:
            vid = reps[0]["variant_id"]
            fold_gates[vid] = data.get("fold_gate", {})
            all_reps.extend(reps)

    df_ranked = rank_variants_table(all_reps, cfg.rubric, fold_gates_by_variant=fold_gates)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_ranked.to_csv(out_csv, index=False)
    print(df_ranked.to_string(index=False))
    if args.plot_png:
        render_ranking_plot(df_ranked, args.plot_png)
        print(f"[rank] Saved ranking plot to {args.plot_png}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pfas-bench", description="RPA1163 PFAS Defluorination Triage Pipeline")
    p.add_argument("--config", default="configs/benchmark_v01.yaml", help="Path to YAML benchmark config")
    sub = p.add_subparsers(dest="command", required=True)

    p_lig = sub.add_parser("prep-ligands", help="Parameterize and pre-extend anionic PFAS substrates")
    p_lig.add_argument("--output-dir", default="data/ligands/prepared")
    p_lig.set_defaults(func=cmd_prep_ligands)

    p_tor = sub.add_parser("torsion-audit", help="Run 1D OpenFF Sage relaxed torsion audit on a C4+ PFAS")
    p_tor.add_argument("--substrate", default="PFBA")
    p_tor.add_argument("--step-deg", type=float, default=15.0)
    p_tor.add_argument("--output-csv", default="results/pfba_torsion_audit.csv")
    p_tor.set_defaults(func=cmd_torsion_audit)

    p_run = sub.add_parser("run-complex", help="Run explicit-solvent or OBC2 MD for one variant x substrate pair")
    p_run.add_argument("--variant", required=True, help="Variant ID from config (e.g. V01_WT, V07_I253A_I135V)")
    p_run.add_argument("--substrate", required=True, help="Substrate ID from config (MFA, DFA, PFBA)")
    p_run.add_argument("--assembly", choices=["chain_A_monomer", "asymmetric_homodimer_AB"], default="chain_A_monomer")
    p_run.add_argument("--solvent", choices=["explicit_tip3pfb", "implicit_obc2"], default="explicit_tip3pfb")
    p_run.add_argument("--replicates", type=int, default=1)
    p_run.add_argument("--equil-steps", type=int, default=500)
    p_run.add_argument("--prod-steps", type=int, default=2000)
    p_run.add_argument("--report-interval", type=int, default=100)
    p_run.add_argument("--temperature-k", type=float, default=300.0)
    p_run.add_argument("--ph", type=float, default=7.5)
    p_run.add_argument("--cacl2-mm", type=float, default=0.0)
    p_run.add_argument("--outer-restraint-k", type=float, default=0.5)
    p_run.add_argument("--output-dir", default="results/runs")
    p_run.set_defaults(func=cmd_run_complex)

    p_rk = sub.add_parser("rank", help="Aggregate replicate outputs into Stage-A vetoes + S_total ranking table")
    p_rk.add_argument("--results-dir", default="results/runs")
    p_rk.add_argument("--output-csv", default="results/summary_ranking.csv")
    p_rk.add_argument("--plot-png", default="results/summary_ranking.png")
    p_rk.set_defaults(func=cmd_rank)

    p_pan = sub.add_parser("run-panel", help="Run multi-variant x multi-substrate panel and generate ranking plot")
    p_pan.add_argument("--variants", nargs="*", default=None)
    p_pan.add_argument("--substrates", nargs="*", default=None)
    p_pan.add_argument("--assembly", choices=["chain_A_monomer", "asymmetric_homodimer_AB"], default="chain_A_monomer")
    p_pan.add_argument("--solvent", choices=["explicit_tip3pfb", "implicit_obc2"], default="explicit_tip3pfb")
    p_pan.add_argument("--replicates", type=int, default=2)
    p_pan.add_argument("--equil-steps", type=int, default=500)
    p_pan.add_argument("--prod-steps", type=int, default=2000)
    p_pan.add_argument("--report-interval", type=int, default=50)
    p_pan.add_argument("--temperature-k", type=float, default=300.0)
    p_pan.add_argument("--ph", type=float, default=7.5)
    p_pan.add_argument("--cacl2-mm", type=float, default=0.0)
    p_pan.add_argument("--outer-restraint-k", type=float, default=0.5)
    p_pan.add_argument("--resume", action="store_true", help="Reuse existing completed pair replicate_metrics.json")
    p_pan.add_argument("--summary-prefix", default="summary_ranking", help="Filename stem for output CSV and PNG")
    p_pan.add_argument("--output-dir", default="results/head_to_head_tip3p")
    p_pan.set_defaults(func=cmd_run_panel)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
