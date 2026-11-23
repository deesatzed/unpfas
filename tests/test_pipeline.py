"""End-to-end unit and physics integration tests for pfas-rpa1163-defluor-bench."""

from pathlib import Path
import shutil

import numpy as np
import pytest
from rdkit.Chem import rdMolTransforms

from pfas_bench.config import BenchmarkConfig, RubricSpec
from pfas_bench.place_substrate import place_substrate_in_pocket
from pfas_bench.prep_ligands import prepare_anionic_substrate
from pfas_bench.prep_receptor import (
    build_receptor_variant,
    evaluate_fold_sanity_gate,
    extract_3r3v_reference,
)
from pfas_bench.run_md import run_holo_replicate
from pfas_bench.score_and_plot import rank_variants_table

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_YAML = ROOT_DIR / "configs/benchmark_v01.yaml"


@pytest.fixture(scope="session", autouse=True)
def ensure_crystal_structures() -> None:
    """Verifies 3R3V and 3R3U mmCIF files are present in data/structures/."""
    dest_dir = ROOT_DIR / "data/structures"
    for fname in ("pdb_00003r3v.cif", "pdb_00003r3u.cif"):
        assert (dest_dir / fname).exists(), f"Missing crystal structure {dest_dir / fname}"


def test_config_and_zone0_immutability_guardrail(tmp_path: Path) -> None:
    cfg = BenchmarkConfig.from_yaml(CONFIG_YAML, root_dir=ROOT_DIR)
    assert len(cfg.variants) >= 10
    assert set(cfg.substrates.keys()) == {"MFA", "DFA", "PFBA"}
    assert 110 in cfg.zone0_residues and 155 in cfg.zone0_residues and 280 in cfg.zone0_residues

    # Verify mutating Zone-0 without negative_control role raises ValueError
    bad_yaml = tmp_path / "bad.yaml"
    txt = CONFIG_YAML.read_text(encoding="utf-8").replace(
        "role: primary_v2_alcove_carve", "role: bad_mut"
    ).replace("{253: \"ALA\"}", "{155: \"ALA\"}")
    bad_yaml.write_text(txt, encoding="utf-8")
    with pytest.raises(ValueError, match="Zone-0"):
        BenchmarkConfig.from_yaml(bad_yaml, root_dir=ROOT_DIR)


def test_ligand_nagl_charging_and_pre_extension() -> None:
    cfg = BenchmarkConfig.from_yaml(CONFIG_YAML, root_dir=ROOT_DIR)
    prep_pfba = prepare_anionic_substrate(cfg.substrates["PFBA"], cfg.charge_method_priority)
    assert abs(prep_pfba.net_charge_e - (-1.0)) < 1e-3
    assert "am1bcc" in prep_pfba.charge_method

    # Confirm PFBA C1-Ca-Cb-Cg starts extended (> 145 deg), NOT curled at 60 deg
    c_idxs = [a.GetIdx() for a in prep_pfba.rd_mol.GetAtoms() if a.GetSymbol() == "C"]
    phi = abs(rdMolTransforms.GetDihedralDeg(prep_pfba.rd_mol.GetConformer(0), *c_idxs[:4]))
    assert phi >= 145.0, f"Expected extended PFBA seed dihedral >= 145 deg, got {phi:.1f} deg"


def test_3r3v_reference_and_pfba_alcove_placement() -> None:
    cfg = BenchmarkConfig.from_yaml(CONFIG_YAML, root_dir=ROOT_DIR)
    ref = extract_3r3v_reference(cfg.cif_holo_path, zone0_seqs=cfg.zone0_residues)
    assert "CH3" in ref.fah_atoms and "F" in ref.fah_atoms
    assert abs(np.linalg.norm(ref.v2_alcove) - 1.0) < 1e-6

    prep_pfba = prepare_anionic_substrate(cfg.substrates["PFBA"], cfg.charge_method_priority)
    placed = place_substrate_in_pocket(prep_pfba.rd_mol, ref, torsion_step_deg=15.0)
    assert placed.selected_torsion_deg is not None
    assert 150.0 <= placed.selected_torsion_deg <= 210.0
    assert placed.min_zone0_clearance_A >= 1.90


def test_explicit_tip3p_holo_md_and_rubric_ranking(tmp_path: Path) -> None:
    """Runs a real explicit-water (TIP3P-FB + PME + NaCl + CaCl2) micro-burst on V07_I253A_I135V + PFBA and V02_D110N."""
    cfg = BenchmarkConfig.from_yaml(CONFIG_YAML, root_dir=ROOT_DIR)
    ref = extract_3r3v_reference(cfg.cif_holo_path, zone0_seqs=cfg.zone0_residues)
    prep_pfba = prepare_anionic_substrate(cfg.substrates["PFBA"], cfg.charge_method_priority)
    placed_pfba = place_substrate_in_pocket(prep_pfba.rd_mol, ref, torsion_step_deg=15.0)

    # 1. Engineered double alcove variant V07 (I253A/I135V) in explicit TIP3P-FB + 10 mM CaCl2
    rec_v07_pdb = tmp_path / "rec_v07.pdb"
    rec_v07 = build_receptor_variant(
        cfg.cif_holo_path, ref, cfg.variants["V07_I253A_I135V"], assembly="chain_A_monomer", ph=7.5, out_pdb=rec_v07_pdb
    )
    fg_v07 = evaluate_fold_sanity_gate(rec_v07_pdb, ref, cfg.rubric, is_experimental_or_homology=True)
    assert fg_v07["stage_a_fold_pass"] is True
    assert fg_v07["zone0_heavy_rmsd_A"] < 0.50

    sum_v07 = run_holo_replicate(
        receptor=rec_v07,
        ligand=prep_pfba,
        placed=placed_pfba,
        ref=ref,
        rubric=cfg.rubric,
        out_dir=tmp_path / "v07_pfba",
        replicate_id=1,
        solvent_mode="explicit_tip3pfb",
        divalent_CaCl2_mM=10.0,
        equilibration_steps=100,
        production_steps=300,
        report_interval=100,
    )
    assert sum_v07["has_explicit_water"] is True
    assert 0.0 <= sum_v07["f_wat_step2"] <= 1.0

    # 2. Negative control V02_D110N must strictly yield p_nac_pct == 0.0%
    rec_v02 = build_receptor_variant(
        cfg.cif_holo_path, ref, cfg.variants["V02_D110N"], assembly="chain_A_monomer", ph=5.5
    )
    sum_v02 = run_holo_replicate(
        receptor=rec_v02,
        ligand=prep_pfba,
        placed=placed_pfba,
        ref=ref,
        rubric=cfg.rubric,
        out_dir=tmp_path / "v02_pfba",
        replicate_id=1,
        solvent_mode="implicit_obc2",
        equilibration_steps=50,
        production_steps=150,
        report_interval=50,
    )
    assert sum_v02["p_nac_pct"] == 0.0

    # 3. Verify Stage-A hard veto zeroes S_total when inverted_tail > 0.50
    vetoed_mock = dict(sum_v07)
    vetoed_mock["variant_id"] = "MOCK_INVERTED"
    vetoed_mock["f_inverted_tail"] = 0.85

    ranked = rank_variants_table(
        [sum_v07, sum_v02, vetoed_mock],
        RubricSpec(),
        fold_gates_by_variant={"V07_I253A_I135V": fg_v07},
    )
    inv_row = ranked[ranked["variant_id"] == "MOCK_INVERTED"].iloc[0]
    assert bool(inv_row["stage_a_pass"]) is False
    assert float(inv_row["S_total"]) == 0.0
