"""Stereoselective crystallographic template alignment and Zone-0 torsion clearance."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms

from pfas_bench.prep_receptor import CrystalReference3R3V


@dataclass
class PlacedSubstrate:
    rd_mol: Chem.Mol
    c1_idx: int
    o1_idx: int
    o2_idx: int
    ca_idx: int
    f_scissile_idx: int
    f_alpha_indices: List[int]
    tail_f_indices: List[int]
    selected_torsion_deg: float | None
    min_zone0_clearance_A: float


def place_substrate_in_pocket(
    rd_mol: Chem.Mol,
    ref: CrystalReference3R3V,
    torsion_step_deg: float = 15.0,
) -> PlacedSubstrate:
    """Aligns carboxylate + C_alpha + scissile F onto FAH 305 and sweeps tail torsion away from Zone 0."""
    mol = Chem.Mol(rd_mol)
    c1_idx, o_idxs, ca_idx = None, [], None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":
            o_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
            if len(o_nbrs) == 2:
                c1_idx = atom.GetIdx()
                o_idxs = o_nbrs
                ca_idx = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "C"][0]
                break

    if c1_idx is None or ca_idx is None:
        raise ValueError("Substrate must contain an alpha-carbon bonded to a carboxylate (-COO-) headgroup.")

    ca_atom = mol.GetAtomWithIdx(ca_idx)
    f_alpha_idxs = [n.GetIdx() for n in ca_atom.GetNeighbors() if n.GetSymbol() == "F"]
    if not f_alpha_idxs:
        raise ValueError("Substrate must contain at least one alpha-fluorine on C_alpha.")

    cb_idxs = [n.GetIdx() for n in ca_atom.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != c1_idx]
    fah = ref.fah_atoms
    target_pts = [fah["C"], fah["O"], fah["OXT"], fah["CH3"], fah["F"]]

    best_result: PlacedSubstrate | None = None
    best_score = -1e12

    for o_perm in [(o_idxs[0], o_idxs[1]), (o_idxs[1], o_idxs[0])]:
        for f_sc in f_alpha_idxs:
            base_trial = Chem.Mol(mol)
            prb_indices = [c1_idx, o_perm[0], o_perm[1], ca_idx, f_sc]

            ref_mol = Chem.RWMol()
            for _ in range(len(prb_indices)):
                ref_mol.AddAtom(Chem.Atom(6))
            ref_conf = Chem.Conformer(len(prb_indices))
            for k, pt in enumerate(target_pts):
                ref_conf.SetAtomPosition(k, pt.tolist())
            ref_mol.AddConformer(ref_conf)

            cg_idx = None
            fg_idx = None
            if cb_idxs:
                cb_atom = base_trial.GetAtomWithIdx(cb_idxs[0])
                cg_cands = [n.GetIdx() for n in cb_atom.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != ca_idx]
                cg_idx = cg_cands[0] if cg_cands else None
                if cg_idx is not None:
                    cg_atom = base_trial.GetAtomWithIdx(cg_idx)
                    fg_cands = [n.GetIdx() for n in cg_atom.GetNeighbors() if n.GetSymbol() == "F"]
                    fg_idx = fg_cands[0] if fg_cands else None

            torsion_grid = np.arange(0.0, 360.0, torsion_step_deg) if (cb_idxs and cg_idx is not None) else [None]
            cf3_staggers = (0.0, 40.0, 80.0) if fg_idx is not None else (None,)

            for tor_deg in torsion_grid:
                for cf3_deg in cf3_staggers:
                    trial = Chem.Mol(base_trial)
                    if tor_deg is not None and cg_idx is not None:
                        rdMolTransforms.SetDihedralDeg(
                            trial.GetConformer(0), c1_idx, ca_idx, cb_idxs[0], cg_idx, float(tor_deg)
                        )
                    if cf3_deg is not None and cg_idx is not None and fg_idx is not None:
                        rdMolTransforms.SetDihedralDeg(
                            trial.GetConformer(0), ca_idx, cb_idxs[0], cg_idx, fg_idx, float(cf3_deg)
                        )

                    rmsd = rdMolAlign.AlignMol(
                        trial, ref_mol, atomMap=[(prb_indices[k], k) for k in range(len(prb_indices))]
                    )
                    t_conf = trial.GetConformer(0)
                    ca_pos = np.array(t_conf.GetAtomPosition(ca_idx))
                    score = -2.0 * rmsd
                    min_z0_dist = 9.99

                    if cb_idxs:
                        cb_pos = np.array(t_conf.GetAtomPosition(cb_idxs[0]))
                        u_cb = (cb_pos - ca_pos) / np.linalg.norm(cb_pos - ca_pos)
                        score += 12.0 * float(np.dot(u_cb, ref.v2_alcove))

                        tail_atoms = cb_idxs + ([cg_idx] if cg_idx is not None else []) + [
                            a.GetIdx() for a in trial.GetAtoms() if a.GetSymbol() == "F" and a.GetIdx() not in f_alpha_idxs
                        ]
                        tail_pts = np.array([t_conf.GetAtomPosition(ti) for ti in tail_atoms])
                        dmat = np.linalg.norm(tail_pts[:, None, :] - ref.zone0_heavy_coords_A[None, :, :], axis=-1)
                        min_z0_dist = float(np.min(dmat))
                        score += 4.5 * min_z0_dist

                        if tor_deg is not None and 155.0 <= float(tor_deg) <= 205.0:
                            score += 1.0
                    elif len(f_alpha_idxs) == 2:
                        f_other = [x for x in f_alpha_idxs if x != f_sc][0]
                        fo_pos = np.array(t_conf.GetAtomPosition(f_other))
                        u_fo = (fo_pos - ca_pos) / np.linalg.norm(fo_pos - ca_pos)
                        score += 6.0 * float(np.dot(u_fo, ref.v2_alcove))
                        dmat = np.linalg.norm(fo_pos[None, :] - ref.zone0_heavy_coords_A, axis=-1)
                        min_z0_dist = float(np.min(dmat))

                    if score > best_score:
                        best_score = score
                        tail_fs = [a.GetIdx() for a in trial.GetAtoms() if a.GetSymbol() == "F" and a.GetIdx() not in f_alpha_idxs]
                        best_result = PlacedSubstrate(
                            rd_mol=trial,
                            c1_idx=c1_idx,
                            o1_idx=o_perm[0],
                            o2_idx=o_perm[1],
                            ca_idx=ca_idx,
                            f_scissile_idx=f_sc,
                            f_alpha_indices=list(f_alpha_idxs),
                            tail_f_indices=tail_fs,
                            selected_torsion_deg=float(tor_deg) if tor_deg is not None else None,
                            min_zone0_clearance_A=round(min_z0_dist, 3),
                        )

    assert best_result is not None
    return best_result
