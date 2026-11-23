"""Trajectory frame analysis: 4-criteria NAC, PBC-aware E_pull field, Step-2 water check, and Stage-A2 vetoes."""

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as ounit

from pfas_bench.config import RubricSpec
from pfas_bench.place_substrate import PlacedSubstrate

KE_MV_CM_PER_E_ANG2 = 1439.9645  # Converts e / Angstrom^2 electric field to MV/cm


@dataclass
class HoloIndexMap:
    has_anionic_asp110: bool
    asp110_attack_O_indices: List[int]
    asp110_cg_idx: int
    his280_ne2_idx: int
    cradle_H_indices: List[int]
    cradle_heavy_indices: List[int]
    clamp_N_indices: List[int]
    water_O_indices: np.ndarray
    env_charge_indices: np.ndarray
    env_charges_e: np.ndarray
    lig_ca_idx: int
    lig_o1_idx: int
    lig_o2_idx: int
    lig_f_alpha_indices: List[int]
    lig_f_tail_indices: List[int]


def build_holo_index_map(
    topology: app.Topology,
    system: openmm.System,
    catalytic_res_indices: Dict[int, int],
    lig_offset: int,
    placed: PlacedSubstrate,
) -> HoloIndexMap:
    """Indexes catalytic protein atoms, water oxygens, environment charges, and ligand atoms."""
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]

    r110 = catalytic_res_indices[110]
    r111 = catalytic_res_indices[111]
    r114 = catalytic_res_indices[114]
    r155 = catalytic_res_indices[155]
    r156 = catalytic_res_indices[156]
    r219 = catalytic_res_indices[219]
    r280 = catalytic_res_indices[280]

    has_anionic_asp110 = False
    asp110_O: List[int] = []
    asp110_cg = -1
    his280_ne2 = -1
    cradle_H: List[int] = []
    cradle_heavy: List[int] = []
    clamp_N: List[int] = []
    water_O: List[int] = []
    env_indices: List[int] = []
    env_charges: List[float] = []

    lig_atom_end = lig_offset + placed.rd_mol.GetNumAtoms()

    for atom in topology.atoms():
        idx = atom.index
        if lig_offset <= idx < lig_atom_end:
            continue

        q_e = nb.getParticleParameters(idx)[0].value_in_unit(ounit.elementary_charge)
        if abs(q_e) > 1e-7:
            env_indices.append(idx)
            env_charges.append(float(q_e))

        rname = atom.residue.name
        aname = atom.name
        ridx = atom.residue.index

        if rname in ("HOH", "WAT", "SOL") and aname == "O":
            water_O.append(idx)
            continue

        if ridx == r110:
            if rname == "ASP":
                has_anionic_asp110 = True
            if aname in ("OD1", "OD2"):
                asp110_O.append(idx)
            elif aname == "CG":
                asp110_cg = idx
        elif ridx == r280 and aname == "NE2":
            his280_ne2 = idx
        elif ridx == r155:
            if aname == "HE2":
                cradle_H.append(idx)
            elif aname == "NE2":
                cradle_heavy.append(idx)
        elif ridx == r156:
            if aname == "HE1":
                cradle_H.append(idx)
            elif aname == "NE1":
                cradle_heavy.append(idx)
        elif ridx == r219:
            if aname == "HH":
                cradle_H.append(idx)
            elif aname == "OH":
                cradle_heavy.append(idx)
        elif ridx in (r111, r114) and aname in ("NE", "NH1", "NH2"):
            clamp_N.append(idx)

    return HoloIndexMap(
        has_anionic_asp110=has_anionic_asp110,
        asp110_attack_O_indices=asp110_O,
        asp110_cg_idx=asp110_cg,
        his280_ne2_idx=his280_ne2,
        cradle_H_indices=cradle_H,
        cradle_heavy_indices=cradle_heavy,
        clamp_N_indices=clamp_N,
        water_O_indices=np.array(water_O, dtype=int),
        env_charge_indices=np.array(env_indices, dtype=int),
        env_charges_e=np.array(env_charges, dtype=float),
        lig_ca_idx=lig_offset + placed.ca_idx,
        lig_o1_idx=lig_offset + placed.o1_idx,
        lig_o2_idx=lig_offset + placed.o2_idx,
        lig_f_alpha_indices=[lig_offset + i for i in placed.f_alpha_indices],
        lig_f_tail_indices=[lig_offset + i for i in placed.tail_f_indices],
    )


def _mic_disp(vec_A: np.ndarray, box_diag_A: np.ndarray | None) -> np.ndarray:
    """Applies orthorhombic minimum-image convention to displacement vector(s) in Angstroms."""
    if box_diag_A is None:
        return vec_A
    return vec_A - box_diag_A * np.round(vec_A / box_diag_A)


def evaluate_snapshot(
    pos_A: np.ndarray,
    imap: HoloIndexMap,
    rubric: RubricSpec,
    box_diag_A: np.ndarray | None = None,
    efield_cutoff_A: float = 15.0,
) -> Dict[str, float | bool]:
    """Computes NAC criteria, scissile E_pull (MV/cm), Step-2 water access, and dead-end flags on one frame."""
    r_ca = pos_A[imap.lig_ca_idx]

    # 1. Carboxylate clamp min distance
    clamp_dists = []
    for in_idx in imap.clamp_N_indices:
        for io_idx in (imap.lig_o1_idx, imap.lig_o2_idx):
            dv = _mic_disp(pos_A[in_idx] - pos_A[io_idx], box_diag_A)
            clamp_dists.append(float(np.linalg.norm(dv)))
    d_clamp = min(clamp_dists) if clamp_dists else 99.9

    # 2. Evaluate all candidate alpha-fluorines simultaneously
    best_tuple = None
    for idx_f in imap.lig_f_alpha_indices:
        r_f = r_ca + _mic_disp(pos_A[idx_f] - r_ca, box_diag_A)
        n_hb = 0
        n_hb_ts = 0
        for ih in imap.cradle_H_indices:
            dh = float(np.linalg.norm(_mic_disp(pos_A[ih] - r_f, box_diag_A)))
            if dh <= rubric.max_cradle_H_F_dist_A:
                n_hb += 1
            if dh <= rubric.ts_tight_H_F_dist_A:
                n_hb_ts += 1

        od_candidates = []
        for iod in imap.asp110_attack_O_indices:
            v_od = _mic_disp(pos_A[iod] - r_ca, box_diag_A)
            od_candidates.append((float(np.linalg.norm(v_od)), v_od))
        d_att, v_od_best = min(od_candidates, key=lambda item: item[0])

        u_od = v_od_best / np.linalg.norm(v_od_best)
        u_f = (r_f - r_ca) / np.linalg.norm(r_f - r_ca)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u_od, u_f))))))

        # Ground-state Michaelis NAC (calibrated to 3R3V where Tyr219 H-bonds to -COO- at 2.74 A)
        is_nac = (
            imap.has_anionic_asp110
            and (d_att <= rubric.max_attack_dist_A)
            and (rubric.min_attack_angle_deg <= ang <= rubric.max_attack_angle_deg)
            and (n_hb >= rubric.min_cradle_hbonds)
            and (d_clamp <= rubric.max_clamp_N_O_dist_A)
        )
        # Tight transition-state-like multi-donor halide compression diagnostic
        is_ts_tight = (
            imap.has_anionic_asp110
            and (d_att <= rubric.ts_tight_attack_dist_A)
            and (rubric.ts_tight_angle_deg <= ang <= rubric.max_attack_angle_deg)
            and (n_hb_ts >= rubric.ts_tight_min_hbonds)
            and (d_clamp <= 3.40)
        )
        is_near_nac = (
            imap.has_anionic_asp110
            and (d_att <= rubric.max_attack_dist_A + 0.15)
            and (135.0 <= ang <= 180.0)
            and (n_hb >= 1)
            and (d_clamp <= rubric.max_clamp_N_O_dist_A + 0.15)
        )
        rank_key = (int(is_ts_tight), int(is_nac), int(is_near_nac), n_hb, ang, -d_att)
        if best_tuple is None or rank_key > best_tuple[0]:
            best_tuple = (rank_key, r_f, d_att, ang, n_hb, is_nac, is_ts_tight, is_near_nac)

    assert best_tuple is not None
    _, r_f_sc, d_att, ang_att, n_hb, is_nac, is_ts_tight, is_near_nac = best_tuple

    # 3. Scissile C-F midpoint electric field projection (E_pull = -E . u_{Ca->F} in MV/cm)
    r_mid = 0.5 * (r_ca + r_f_sc)
    u_cf = (r_f_sc - r_ca) / np.linalg.norm(r_f_sc - r_ca)
    diff = _mic_disp(r_mid[None, :] - pos_A[imap.env_charge_indices], box_diag_A)
    r_norm = np.linalg.norm(diff, axis=1)
    mask = (r_norm > 0.5) & (r_norm <= efield_cutoff_A)
    if np.any(mask):
        d_sub = diff[mask]
        r_sub = r_norm[mask]
        q_sub = imap.env_charges_e[mask]
        e_vec = KE_MV_CM_PER_E_ANG2 * np.sum((q_sub[:, None] * d_sub) / (r_sub[:, None] ** 3), axis=0)
        e_pull = float(-np.dot(e_vec, u_cf))
    else:
        e_pull = 0.0

    # 4. Step-2 hydrolytic water check near His280:NE2 and Asp110:CG
    if len(imap.water_O_indices) > 0 and imap.his280_ne2_idx >= 0 and imap.asp110_cg_idx >= 0:
        wat_pts = pos_A[imap.water_O_indices]
        d_his = np.linalg.norm(_mic_disp(wat_pts - pos_A[imap.his280_ne2_idx][None, :], box_diag_A), axis=1)
        d_cg = np.linalg.norm(_mic_disp(wat_pts - pos_A[imap.asp110_cg_idx][None, :], box_diag_A), axis=1)
        step2_water_ok = bool(
            np.any(
                (d_his <= rubric.max_His280_NE2_water_O_dist_A)
                & (d_cg <= rubric.max_Asp110_CG_water_O_dist_A)
            )
        )
    else:
        # Implicit-solvent runs lack discrete water oxygens
        step2_water_ok = True

    # 5. Stage-A2 dead-end pose checks
    cradle_pts = np.array([r_ca + _mic_disp(pos_A[i] - r_ca, box_diag_A) for i in imap.cradle_heavy_indices])
    cradle_center = np.mean(cradle_pts, axis=0)
    d_alpha_cradle = min(
        float(np.linalg.norm((r_ca + _mic_disp(pos_A[i] - r_ca, box_diag_A)) - cradle_center))
        for i in imap.lig_f_alpha_indices
    )
    if imap.lig_f_tail_indices:
        d_tail_cradle = min(
            float(np.linalg.norm((r_ca + _mic_disp(pos_A[i] - r_ca, box_diag_A)) - cradle_center))
            for i in imap.lig_f_tail_indices
        )
        inverted = d_tail_cradle < d_alpha_cradle
    else:
        inverted = False

    clamp_unbound = d_clamp > 4.50

    return {
        "d_attack_A": d_att,
        "ang_SN2_deg": ang_att,
        "cradle_hbonds": float(n_hb),
        "d_clamp_A": d_clamp,
        "is_nac": float(is_nac),
        "is_ts_tight_nac": float(is_ts_tight),
        "is_near_nac": float(is_near_nac),
        "E_pull_MV_cm": e_pull,
        "step2_water_ok": float(step2_water_ok),
        "inverted_pose": float(inverted),
        "clamp_unbound": float(clamp_unbound),
    }


def summarize_trajectory_frames(
    frame_records: Sequence[Dict[str, float | bool]],
    ca_coords_A: np.ndarray,
    rubric: RubricSpec,
    has_explicit_water: bool = True,
    has_anionic_asp110: bool = True,
) -> Dict[str, float | bool]:
    """Aggregates per-frame metrics into production trajectory observables and Stage-A2 veto status."""
    if not frame_records:
        raise ValueError("No trajectory frames recorded for summary.")

    p_nac = float(np.mean([r["is_nac"] for r in frame_records]))
    p_ts_tight = float(np.mean([r.get("is_ts_tight_nac", 0.0) for r in frame_records]))
    p_near_nac = float(np.mean([r.get("is_near_nac", r["is_nac"]) for r in frame_records]))
    nac_e = [float(r["E_pull_MV_cm"]) for r in frame_records if r["is_nac"] > 0.5]
    all_e = [float(r["E_pull_MV_cm"]) for r in frame_records]
    e_proj_mean = float(np.mean(nac_e)) if nac_e else float(np.mean(all_e))

    f_wat_step2 = float(np.mean([r["step2_water_ok"] for r in frame_records]))
    f_inverted = float(np.mean([r["inverted_pose"] for r in frame_records]))
    f_unbound = float(np.mean([r["clamp_unbound"] for r in frame_records]))

    mean_ca = np.mean(ca_coords_A, axis=0)
    ca_rmsf = float(np.sqrt(np.mean(np.sum((ca_coords_A - mean_ca) ** 2, axis=1))))

    stage_a_pose_pass = (
        bool(has_anionic_asp110)
        and (f_inverted <= rubric.max_inverted_tail_in_cradle_fraction)
        and (f_unbound <= rubric.max_clamp_unbound_fraction)
    )

    return {
        "n_frames": len(frame_records),
        "has_explicit_water": bool(has_explicit_water),
        "has_anionic_asp110": bool(has_anionic_asp110),
        "p_nac_fraction": round(p_nac, 4),
        "p_nac_pct": round(100.0 * p_nac, 2),
        "p_ts_tight_nac_pct": round(100.0 * p_ts_tight, 2),
        "p_near_nac_pct": round(100.0 * p_near_nac, 2),
        "mean_E_pull_MV_cm": round(float(np.mean(all_e)), 2),
        "nac_E_pull_MV_cm": round(e_proj_mean, 2),
        "mean_d_attack_A": round(float(np.mean([r["d_attack_A"] for r in frame_records])), 3),
        "min_d_attack_A": round(float(np.min([r["d_attack_A"] for r in frame_records])), 3),
        "mean_ang_SN2_deg": round(float(np.mean([r["ang_SN2_deg"] for r in frame_records])), 2),
        "max_ang_SN2_deg": round(float(np.max([r["ang_SN2_deg"] for r in frame_records])), 2),
        "mean_cradle_hbonds": round(float(np.mean([r["cradle_hbonds"] for r in frame_records])), 3),
        "mean_d_clamp_A": round(float(np.mean([r["d_clamp_A"] for r in frame_records])), 3),
        "ca_rmsf_A": round(ca_rmsf, 3),
        "f_wat_step2": round(f_wat_step2, 4),
        "f_inverted_tail": round(f_inverted, 4),
        "f_clamp_unbound": round(f_unbound, 4),
        "stage_a_pose_pass": bool(stage_a_pose_pass),
    }
