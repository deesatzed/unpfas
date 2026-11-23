"""OpenMM explicit-solvent (TIP3P-FB + PME + NaCl/CaCl2) and OBC2 MD engine."""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as ounit
from openff.toolkit.typing.engines.smirnoff import ForceField as OffForceField

from pfas_bench.analyze_nac_efield import (
    _mic_disp,
    build_holo_index_map,
    evaluate_snapshot,
    summarize_trajectory_frames,
)
from pfas_bench.config import RubricSpec
from pfas_bench.place_substrate import PlacedSubstrate
from pfas_bench.prep_ligands import PreparedLigand, attach_obc2_solvent
from pfas_bench.prep_receptor import CrystalReference3R3V, PreparedReceptor


def _get_force(system: openmm.System, cls: type):
    matches = [f for f in system.getForces() if isinstance(f, cls)]
    return matches[0] if matches else None


def build_holo_openmm_system(
    receptor: PreparedReceptor,
    ligand: PreparedLigand,
    placed: PlacedSubstrate,
    ref: CrystalReference3R3V,
    solvent_mode: str = "explicit_tip3pfb",
    box_padding_nm: float = 1.0,
    nonbonded_cutoff_nm: float = 0.9,
    ionic_strength_M: float = 0.15,
    divalent_CaCl2_mM: float = 0.0,
    outer_restraint_kcal_mol_A2: float = 0.5,
    outer_restraint_cutoff_A: float = 15.0,
    forcefield_lig: str = "openff-2.2.0.offxml",
) -> Tuple[app.Topology, List[openmm.Vec3], openmm.System, int, bool]:
    """Constructs an explicit-solvent PME or implicit-OBC2 holo System merging Amber14SB + OpenFF Sage 2.2.0."""
    conf = placed.rd_mol.GetConformer(0)
    lig_coords_A = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(placed.rd_mol.GetNumAtoms())],
        dtype=float,
    )

    ff_prot = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = app.Modeller(receptor.topology, receptor.positions)
    has_explicit_water = solvent_mode == "explicit_tip3pfb"

    if has_explicit_water:
        modeller.addSolvent(
            ff_prot,
            model="tip3p",
            padding=box_padding_nm * ounit.nanometer,
            ionicStrength=ionic_strength_M * ounit.molar,
            neutralize=True,
        )
        # Carve waters overlapping placed ligand (< 2.55 A) or mutated protein heavy atoms (< 2.15 A)
        pos_solv_A = np.array(
            [[p.value_in_unit(ounit.angstrom)[0], p.value_in_unit(ounit.angstrom)[1], p.value_in_unit(ounit.angstrom)[2]] for p in modeller.positions],
            dtype=float,
        )
        prot_heavy_pts = np.array(
            [
                pos_solv_A[a.index]
                for a in modeller.topology.atoms()
                if a.residue.name not in ("HOH", "WAT", "NA", "CL", "CA") and a.element.atomic_number > 1
            ],
            dtype=float,
        )
        to_delete = []
        for res in modeller.topology.residues():
            if res.name in ("HOH", "WAT"):
                res_pts = np.array([pos_solv_A[a.index] for a in res.atoms()])
                dmin_lig = float(np.min(np.linalg.norm(res_pts[:, None, :] - lig_coords_A[None, :, :], axis=-1)))
                o_pts = np.array([pos_solv_A[a.index] for a in res.atoms() if a.name == "O"])
                dmin_prot = float(np.min(np.linalg.norm(o_pts[:, None, :] - prot_heavy_pts[None, :, :], axis=-1)))
                if dmin_lig < 2.55 or dmin_prot < 2.15:
                    to_delete.append(res)
        if to_delete:
            modeller.delete(to_delete)

        # Optional CaCl2 groundwater matrix doping: swap distant water molecules (> 12 A from ligand) into CA + 2 CL
        if divalent_CaCl2_mM > 0.0:
            box_vecs = modeller.topology.getPeriodicBoxVectors()
            vol_nm3 = float(
                box_vecs[0][0].value_in_unit(ounit.nanometer)
                * box_vecs[1][1].value_in_unit(ounit.nanometer)
                * box_vecs[2][2].value_in_unit(ounit.nanometer)
            )
            n_ca = max(1, int(round(divalent_CaCl2_mM * 1e-3 * 0.602214 * vol_nm3)))
            pos_cur_A = np.array(
                [[p.value_in_unit(ounit.angstrom)[0], p.value_in_unit(ounit.angstrom)[1], p.value_in_unit(ounit.angstrom)[2]] for p in modeller.positions]
            )
            far_waters = []
            for res in modeller.topology.residues():
                if res.name == "HOH":
                    o_atom = [a for a in res.atoms() if a.name == "O"][0]
                    if float(np.linalg.norm(pos_cur_A[o_atom.index] - ref.fah_atoms["CH3"])) > 14.0:
                        far_waters.append((res, pos_cur_A[o_atom.index]))
            if len(far_waters) >= 3 * n_ca:
                chosen = far_waters[: 3 * n_ca]
                modeller.delete([item[0] for item in chosen])
                ion_top = app.Topology()
                ion_chain = ion_top.addChain("I")
                ion_pos = []
                for k in range(n_ca):
                    r_ca_res = ion_top.addResidue("CA", ion_chain)
                    ion_top.addAtom("CA", app.Element.getBySymbol("Ca"), r_ca_res)
                    pt_ca = chosen[3 * k][1]
                    ion_pos.append(openmm.Vec3(float(pt_ca[0]), float(pt_ca[1]), float(pt_ca[2])) * ounit.angstrom)
                    for m in (1, 2):
                        r_cl_res = ion_top.addResidue("CL", ion_chain)
                        ion_top.addAtom("CL", app.Element.getBySymbol("Cl"), r_cl_res)
                        pt_cl = chosen[3 * k + m][1]
                        ion_pos.append(openmm.Vec3(float(pt_cl[0]), float(pt_cl[1]), float(pt_cl[2])) * ounit.angstrom)
                modeller.add(ion_top, ion_pos)

        prot_sys = ff_prot.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=nonbonded_cutoff_nm * ounit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
    elif solvent_mode == "implicit_obc2":
        # Strip crystallographic waters when running continuum implicit_obc2
        wat_del = [r for r in modeller.topology.residues() if r.name in ("HOH", "WAT")]
        if wat_del:
            modeller.delete(wat_del)
        prot_sys = app.ForceField("amber14-all.xml").createSystem(
            modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
    else:
        raise ValueError(f"Unknown solvent_mode: {solvent_mode}")

    # Build OpenFF Sage 2.2.0 ligand system and merge into prot_sys
    off_ff = OffForceField(forcefield_lig)
    lig_sys = off_ff.create_openmm_system(
        ligand.off_mol.to_topology(), charge_from_molecules=[ligand.off_mol]
    )

    lig_offset = prot_sys.getNumParticles()
    for i in range(lig_sys.getNumParticles()):
        prot_sys.addParticle(lig_sys.getParticleMass(i))

    # Merge HarmonicBondForce (converting X-H bonds to constraints if HBonds active)
    p_bond = _get_force(prot_sys, openmm.HarmonicBondForce)
    l_bond = _get_force(lig_sys, openmm.HarmonicBondForce)
    lig_atomic_nums = [a.atomic_number for a in ligand.off_mol.atoms]
    if l_bond is not None:
        for idx in range(l_bond.getNumBonds()):
            a1, a2, length, k = l_bond.getBondParameters(idx)
            if lig_atomic_nums[a1] == 1 or lig_atomic_nums[a2] == 1:
                prot_sys.addConstraint(lig_offset + a1, lig_offset + a2, length)
            else:
                p_bond.addBond(lig_offset + a1, lig_offset + a2, length, k)

    p_ang = _get_force(prot_sys, openmm.HarmonicAngleForce)
    l_ang = _get_force(lig_sys, openmm.HarmonicAngleForce)
    if l_ang is not None:
        for idx in range(l_ang.getNumAngles()):
            a1, a2, a3, angle, k = l_ang.getAngleParameters(idx)
            p_ang.addAngle(lig_offset + a1, lig_offset + a2, lig_offset + a3, angle, k)

    p_tor = _get_force(prot_sys, openmm.PeriodicTorsionForce)
    l_tor = _get_force(lig_sys, openmm.PeriodicTorsionForce)
    if l_tor is not None:
        for idx in range(l_tor.getNumTorsions()):
            a1, a2, a3, a4, per, phase, k = l_tor.getTorsionParameters(idx)
            p_tor.addTorsion(lig_offset + a1, lig_offset + a2, lig_offset + a3, lig_offset + a4, per, phase, k)

    p_nb = _get_force(prot_sys, openmm.NonbondedForce)
    l_nb = _get_force(lig_sys, openmm.NonbondedForce)
    for i in range(l_nb.getNumParticles()):
        q, sig, eps = l_nb.getParticleParameters(i)
        p_nb.addParticle(q, sig, eps)
    for j in range(l_nb.getNumExceptions()):
        a1, a2, qprod, sig, eps = l_nb.getExceptionParameters(j)
        p_nb.addException(lig_offset + a1, lig_offset + a2, qprod, sig, eps)

    lig_top = ligand.off_mol.to_topology().to_openmm()
    for res in lig_top.residues():
        res.name = "LIG"
    lig_pos = [openmm.Vec3(float(c[0]), float(c[1]), float(c[2])) * ounit.angstrom for c in lig_coords_A]
    modeller.add(lig_top, lig_pos)

    if not has_explicit_water:
        z_all = [atom.element.atomic_number for atom in modeller.topology.atoms()]
        attach_obc2_solvent(prot_sys, z_all)

    # Outer-shell harmonic restraint (Tier-1 monomer only; set outer_restraint_kcal_mol_A2=0 for unrestrained Tier-2 dimer)
    if outer_restraint_kcal_mol_A2 > 0.0:
        k_kj_nm2 = float(outer_restraint_kcal_mol_A2) * 418.4
        if has_explicit_water:
            expr = "0.5 * k_pos * periodicdistance(x, y, z, x0, y0, z0)^2"
        else:
            expr = "0.5 * k_pos * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
        restraint = openmm.CustomExternalForce(expr)
        restraint.addGlobalParameter("k_pos", k_kj_nm2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        pos_nm = [p.value_in_unit(ounit.nanometer) for p in modeller.positions]
        fah_ca = ref.fah_atoms["CH3"]
        for atom in modeller.topology.atoms():
            if atom.index < lig_offset and atom.name == "CA" and atom.residue.name not in ("CA", "CL", "NA", "HOH"):
                p_A = np.array([pos_nm[atom.index][0], pos_nm[atom.index][1], pos_nm[atom.index][2]]) * 10.0
                if float(np.linalg.norm(p_A - fah_ca)) > outer_restraint_cutoff_A:
                    restraint.addParticle(atom.index, pos_nm[atom.index])
        if restraint.getNumParticles() > 0:
            prot_sys.addForce(restraint)

    return modeller.topology, modeller.positions, prot_sys, lig_offset, has_explicit_water


def run_holo_replicate(
    receptor: PreparedReceptor,
    ligand: PreparedLigand,
    placed: PlacedSubstrate,
    ref: CrystalReference3R3V,
    rubric: RubricSpec,
    out_dir: str | Path,
    replicate_id: int = 1,
    solvent_mode: str = "explicit_tip3pfb",
    temperature_K: float = 300.0,
    ionic_strength_M: float = 0.15,
    divalent_CaCl2_mM: float = 0.0,
    outer_restraint_kcal_mol_A2: float = 0.5,
    outer_restraint_cutoff_A: float = 15.0,
    equilibration_steps: int = 1000,
    production_steps: int = 5000,
    report_interval: int = 100,
    timestep_fs: float = 2.0,
    random_seed: int = 42,
    resume: bool = False,
) -> Dict[str, Any]:
    """Runs minimization, equilibration, and production MD for one replicate with live ETA & reboot-safe XML checkpoints."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    rep_summary_json = out_p / f"rep{replicate_id:02d}_summary.json"
    if resume and rep_summary_json.exists():
        with open(rep_summary_json, "r", encoding="utf-8") as fh:
            cached_rep = json.load(fh)
        print(
            f"   [resume] Loaded finished {receptor.variant.id} + {ligand.spec.id} (rep {replicate_id})",
            flush=True,
        )
        return cached_rep

    top, pos, system, lig_offset, has_explicit_water = build_holo_openmm_system(
        receptor=receptor,
        ligand=ligand,
        placed=placed,
        ref=ref,
        solvent_mode=solvent_mode,
        ionic_strength_M=ionic_strength_M,
        divalent_CaCl2_mM=divalent_CaCl2_mM,
        outer_restraint_kcal_mol_A2=outer_restraint_kcal_mol_A2,
        outer_restraint_cutoff_A=outer_restraint_cutoff_A,
    )

    if has_explicit_water:
        system.addForce(openmm.MonteCarloBarostat(1.0 * ounit.bar, temperature_K * ounit.kelvin, 50))

    imap = build_holo_index_map(top, system, receptor.catalytic_res_indices, lig_offset, placed)

    def _execute_trajectory(use_cpu_fallback: bool = False, seed_offset: int = 0):
        # Use a fixed 1.0 fs step throughout (never mutate stepSize mid-context) for rock-solid C-F / PME stability
        eff_dt_ps = min(float(timestep_fs), 1.0) * 1e-3
        integ = openmm.LangevinMiddleIntegrator(
            temperature_K * ounit.kelvin,
            2.0 / ounit.picoseconds,
            eff_dt_ps * ounit.picoseconds,
        )
        seed_val = int(random_seed + replicate_id + seed_offset)
        integ.setRandomNumberSeed(seed_val)

        if use_cpu_fallback:
            plat = openmm.Platform.getPlatformByName("CPU")
            sim_obj = app.Simulation(top, system, integ, plat)
        else:
            try:
                plat = openmm.Platform.getPlatformByName("OpenCL")
                sim_obj = app.Simulation(top, system, integ, plat, {"Precision": "mixed"})
            except Exception:
                sim_obj = app.Simulation(top, system, integ)

        m_pdb = out_p / f"rep{replicate_id:02d}_minimized.pdb"
        chk_xml = out_p / f"rep{replicate_id:02d}_checkpoint.xml"
        chk_meta = out_p / f"rep{replicate_id:02d}_checkpoint_frames.json"

        n_fr = max(1, int(production_steps) // int(report_interval))
        f_recs: List[Dict[str, float | bool]] = []
        ca_list: List[np.ndarray] = []
        start_fr = 0
        e_min_val = 0.0

        # Portable XML state resume (survives macOS updates, driver changes, and reboots!)
        if resume and chk_xml.exists() and chk_meta.exists() and not use_cpu_fallback:
            with open(chk_meta, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            if int(saved.get("n_fr", -1)) == n_fr and int(saved.get("report_interval", -1)) == int(report_interval):
                sim_obj.loadState(str(chk_xml))
                f_recs = list(saved["frame_records"])
                ca_list = [np.array(pt, dtype=float) for pt in saved["ca_list"]]
                start_fr = len(f_recs)
                e_min_val = float(saved.get("e_min_val", 0.0))
                pct_res = 100.0 * start_fr / n_fr
                print(
                    f"      [resume-mid-clip] Resumed rep {replicate_id} at frame {start_fr}/{n_fr} ({pct_res:.1f}%)",
                    flush=True,
                )

        if start_fr == 0:
            sim_obj.context.setPositions(pos)
            sim_obj.minimizeEnergy(
                tolerance=5.0 * ounit.kilojoules_per_mole / ounit.nanometer,
                maxIterations=5000,
            )
            m_state = sim_obj.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=has_explicit_water)
            e_min_val = m_state.getPotentialEnergy().value_in_unit(ounit.kilocalories_per_mole)

            with open(m_pdb, "w", encoding="utf-8") as fh:
                app.PDBFile.writeFile(top, m_state.getPositions(), fh, keepIds=True)

            sim_obj.context.setVelocitiesToTemperature(0.5 * temperature_K * ounit.kelvin, seed_val)
            sim_obj.step(250)
            sim_obj.context.setVelocitiesToTemperature(temperature_K * ounit.kelvin, seed_val + 17)
            if equilibration_steps > 0:
                sim_obj.step(int(equilibration_steps))

        # Log progress & save reboot-safe checkpoint roughly every 5% (or every >= 90 seconds on long runs)
        log_stride = max(1, n_fr // 20)
        t_loop_start = time.monotonic()
        t_last_log = t_loop_start
        steps_per_frame = int(report_interval)
        total_ns = (n_fr * steps_per_frame * eff_dt_ps) / 1000.0

        for fr_idx in range(start_fr, n_fr):
            sim_obj.step(steps_per_frame)
            st_cur = sim_obj.context.getState(getPositions=True, enforcePeriodicBox=has_explicit_water)
            p_A = st_cur.getPositions(asNumpy=True).value_in_unit(ounit.angstrom)
            b_diag = None
            if has_explicit_water:
                bv = st_cur.getPeriodicBoxVectors(asNumpy=True).value_in_unit(ounit.angstrom)
                b_diag = np.array([bv[0, 0], bv[1, 1], bv[2, 2]], dtype=float)
            f_recs.append(evaluate_snapshot(p_A, imap, rubric, box_diag_A=b_diag))
            rel_ca = _mic_disp(p_A[imap.lig_ca_idx] - p_A[imap.asp110_cg_idx], b_diag)
            ca_list.append(rel_ca.copy())

            done_fr = fr_idx + 1
            now = time.monotonic()
            should_checkpoint = (
                production_steps >= 10000
                and ((done_fr % log_stride == 0) or (now - t_last_log >= 120.0))
                and done_fr < n_fr
            )
            if should_checkpoint:
                t_last_log = now
                elapsed_s = max(1e-3, now - t_loop_start)
                done_since_resume = done_fr - start_fr
                steps_per_s = (done_since_resume * steps_per_frame) / elapsed_s
                ns_per_day = (steps_per_s * eff_dt_ps * 86400.0) / 1000.0
                rem_steps = (n_fr - done_fr) * steps_per_frame
                eta_s = rem_steps / max(1e-3, steps_per_s)
                eta_h, eta_rem = divmod(int(eta_s), 3600)
                eta_m = eta_rem // 60
                eta_str = f"{eta_h}h {eta_m:02d}m" if eta_h > 0 else f"{eta_m}m {eta_rem % 60:02d}s"

                cur_ns = (done_fr * steps_per_frame * eff_dt_ps) / 1000.0
                pct = 100.0 * done_fr / n_fr
                live_nac = 100.0 * float(np.mean([r["is_nac"] for r in f_recs]))
                live_ang = float(np.mean([r["ang_SN2_deg"] for r in f_recs]))
                live_epull = float(np.mean([r["E_pull_MV_cm"] for r in f_recs]))

                # Atomic portable XML checkpoint write
                tmp_xml = out_p / f"rep{replicate_id:02d}_checkpoint.xml.tmp"
                tmp_meta = out_p / f"rep{replicate_id:02d}_checkpoint_frames.json.tmp"
                sim_obj.saveState(str(tmp_xml))
                with open(tmp_meta, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "n_fr": n_fr,
                            "report_interval": steps_per_frame,
                            "e_min_val": e_min_val,
                            "frame_records": f_recs,
                            "ca_list": [c.tolist() for c in ca_list],
                        },
                        fh,
                    )
                os.replace(tmp_xml, chk_xml)
                os.replace(tmp_meta, chk_meta)

                print(
                    f"      [{pct:5.1f}% | {cur_ns:.2f}/{total_ns:.2f} ns] "
                    f"speed={ns_per_day:.1f} ns/day | clip ETA={eta_str} | "
                    f"so far: P_NAC={live_nac:.1f}%, mean_ang={live_ang:.1f}deg, E_pull={live_epull:+.1f} MV/cm",
                    flush=True,
                )

        # Clean up temporary mid-clip checkpoint files once replicate finishes cleanly
        for f_cleanup in (chk_xml, chk_meta):
            if f_cleanup.exists():
                f_cleanup.unlink()

        return e_min_val, m_pdb, f_recs, np.array(ca_list, dtype=float)

    try:
        e_min_kcal, min_pdb, frame_records, ca_arr = _execute_trajectory(use_cpu_fallback=False, seed_offset=0)
    except openmm.OpenMMException:
        e_min_kcal, min_pdb, frame_records, ca_arr = _execute_trajectory(use_cpu_fallback=True, seed_offset=101)
    eff_dt_fs = min(float(timestep_fs), 1.0)
    summary = summarize_trajectory_frames(
        frame_records,
        ca_arr,
        rubric,
        has_explicit_water=has_explicit_water,
        has_anionic_asp110=imap.has_anionic_asp110,
    )
    summary.update({
        "variant_id": receptor.variant.id,
        "substrate_id": ligand.spec.id,
        "replicate_id": int(replicate_id),
        "random_seed": int(random_seed + replicate_id),
        "timestep_fs": round(eff_dt_fs, 3),
        "equil_duration_ps": round(int(equilibration_steps) * eff_dt_fs * 1e-3, 4),
        "sim_duration_ps": round(int(production_steps) * eff_dt_fs * 1e-3, 4),
        "report_interval_steps": int(report_interval),
        "assembly": receptor.assembly,
        "solvent_mode": solvent_mode,
        "pH": receptor.ph,
        "temperature_K": temperature_K,
        "divalent_CaCl2_mM": divalent_CaCl2_mM,
        "charge_method": ligand.charge_method,
        "selected_torsion_deg": placed.selected_torsion_deg,
        "e_min_kcal_mol": round(float(e_min_kcal), 2),
        "minimized_pdb": str(min_pdb),
    })
    tmp_rep = out_p / f"rep{replicate_id:02d}_summary.json.tmp"
    with open(tmp_rep, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp_rep, rep_summary_json)
    return summary
