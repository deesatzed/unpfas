# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cachetools",
#     "gemmi",
#     "lxml",
#     "matplotlib",
#     "networkx",
#     "numpy",
#     "openff-forcefields",
#     "openff-interchange",
#     "openff-toolkit",
#     "openff-units",
#     "openmm",
#     "packaging",
#     "pandas",
#     "pdbfixer",
#     "pydantic",
#     "rdkit",
#     "seaborn",
#     "xmltodict",
# ]
# ///

"""Physics Smoke Test for RPA1163 PFAS Defluorination Triage Pipeline.

Uses OpenFF Sage (openff-2.2.0.offxml) directly via OpenFF Toolkit / Interchange
and OpenMM (Amber14SB + native GBSAOBCForce) without openmmforcefields.

Executes:
1. OpenFF Sage (openff-2.2.0) standalone 10 ps Langevin MD + 1D relaxed torsion scan
   (C1-Ca-Cb-Cg) on PFBA, comparing against MMFF94s.
2. Prepares RPA1163 Chain A from PDB 3R3V in two states:
   - WT (N110D reversion; His155=HIE halide donor, His280=HID catalytic base)
   - I253A (N110D + I253A primary prochiral-v2 alcove carve)
3. Stereoselective crystallographic template alignment onto FAH 305 (locking COO-
   in Arg111/Arg114, scissile F in His155-Trp156-Tyr219, and projecting PFBA's
   -CF2CF3 tail along prochiral alcove vector v2).
4. OpenMM (Amber14SB + OpenFF Sage 2.2.0 + OBC2) energy minimization + 5 ps (300 K)
   active-site thermal sampling across all 6 complexes ({WT, I253A} x {MFA, DFA, PFBA}),
   computing d_attack, theta_SN2, halide-cradle H-bonds, clamp lock, P_NAC, and
   scissile bond polarizing electric field E_pull (MV/cm).
"""

import json
import math
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as ounit
from openff.toolkit.topology import Molecule as OffMolecule
from openff.toolkit.typing.engines.smirnoff import ForceField as OffForceField
from openff.units import unit as off_unit
import pandas as pd
import pdbfixer
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

_RDKIT_GEN_SCRIPTS = "/Users/o2satz/.gemini/config/plugins/science/skills/rdkit_general/scripts"
if _RDKIT_GEN_SCRIPTS not in sys.path:
    sys.path.append(_RDKIT_GEN_SCRIPTS)
from compat import PUBLICATION_CYCLE, apply_publication_spines, set_safe_plot_style

warnings.filterwarnings("ignore")

KE_MV_CM_PER_E_ANG2 = 1439.9645  # Converts e / Angstrom^2 electric field to MV/cm

# Standard Bondi radii (nm) and OBC2 scaling factors by atomic number
_OBC2_PARAMS_BY_Z = {
    1: (0.120, 0.85),   # H
    6: (0.170, 0.72),   # C
    7: (0.155, 0.79),   # N
    8: (0.150, 0.85),   # O
    9: (0.147, 0.88),   # F
    15: (0.185, 0.86),  # P
    16: (0.180, 0.96),  # S
    17: (0.175, 0.80),  # Cl
}


def add_obc2_implicit_solvent(system: openmm.System, atomic_numbers: List[int]) -> None:
    """Attaches a native OpenMM GBSAOBCForce matching NonbondedForce charges and Bondi radii."""
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    gb = openmm.GBSAOBCForce()
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    for idx, z in enumerate(atomic_numbers):
        q, _, _ = nb.getParticleParameters(idx)
        rad_nm, scale = _OBC2_PARAMS_BY_Z.get(z, (0.170, 0.80))
        gb.addParticle(q, rad_nm * ounit.nanometer, scale)
    system.addForce(gb)


def assign_charges_openff(rd_mol: Chem.Mol, name: str) -> Tuple[OffMolecule, str]:
    """Creates an OpenFF Molecule with AM1-BCC (if sqm available) or formal-charge-conserving MMFF94s charges."""
    off_mol = OffMolecule.from_rdkit(rd_mol, allow_undefined_stereo=True)
    off_mol.name = name
    formal_q = float(Chem.GetFormalCharge(rd_mol))

    for method_name in ("am1bcc", "openff-gnn-am1bcc-0.1.0-rc.3.pt", "openff-gnn-am1bcc-0.1.0-rc.2.pt"):
        try:
            off_mol.assign_partial_charges(partial_charge_method=method_name)
            return off_mol, method_name
        except Exception:
            pass

    mp = AllChem.MMFFGetMoleculeProperties(rd_mol, mmffVariant="MMFF94s")
    if mp is not None:
        charges = np.array([mp.GetMMFFPartialCharge(i) for i in range(rd_mol.GetNumAtoms())], dtype=float)
        method = "mmff94s_fallback"
    else:
        AllChem.ComputeGasteigerCharges(rd_mol)
        charges = np.array([float(a.GetProp("_GasteigerCharge")) for a in rd_mol.GetAtoms()], dtype=float)
        charges = np.nan_to_num(charges, nan=0.0)
        method = "gasteiger_fallback"

    drift = (formal_q - float(np.sum(charges))) / len(charges)
    charges = charges + drift
    off_mol.partial_charges = charges * off_unit.elementary_charge
    return off_mol, method


def build_openff_ligand_system(off_mol: OffMolecule, include_obc2: bool = True) -> openmm.System:
    """Builds a standalone OpenMM System for an OpenFF molecule using openff-2.2.0.offxml."""
    off_ff = OffForceField("openff-2.2.0.offxml")
    sys_lig = off_ff.create_openmm_system(
        off_mol.to_topology(),
        charge_from_molecules=[off_mol],
    )
    if include_obc2:
        z_list = [atom.atomic_number for atom in off_mol.atoms]
        add_obc2_implicit_solvent(sys_lig, z_list)
    return sys_lig


def merge_protein_and_openff_ligand_systems(
    prot_top: app.Topology,
    prot_pos: List[openmm.Vec3],
    off_mol: OffMolecule,
    lig_coords_ang: np.ndarray,
) -> Tuple[app.Topology, List[openmm.Vec3], openmm.System, int]:
    """Combines Amber14SB protein System and OpenFF Sage 2.2.0 ligand System cleanly."""
    ff_prot = app.ForceField("amber14-all.xml")
    prot_sys = ff_prot.createSystem(prot_top, nonbondedMethod=app.NoCutoff, constraints=None)
    lig_sys = build_openff_ligand_system(off_mol, include_obc2=False)

    lig_offset = prot_sys.getNumParticles()
    for i in range(lig_sys.getNumParticles()):
        prot_sys.addParticle(lig_sys.getParticleMass(i))

    # Helper to find or create standard OpenMM forces on prot_sys
    def get_force(sys_obj, cls):
        matches = [f for f in sys_obj.getForces() if isinstance(f, cls)]
        return matches[0] if matches else None

    # 1. HarmonicBondForce
    p_bond = get_force(prot_sys, openmm.HarmonicBondForce)
    l_bond = get_force(lig_sys, openmm.HarmonicBondForce)
    if l_bond is not None:
        for idx in range(l_bond.getNumBonds()):
            a1, a2, length, k = l_bond.getBondParameters(idx)
            p_bond.addBond(lig_offset + a1, lig_offset + a2, length, k)

    # 2. HarmonicAngleForce
    p_ang = get_force(prot_sys, openmm.HarmonicAngleForce)
    l_ang = get_force(lig_sys, openmm.HarmonicAngleForce)
    if l_ang is not None:
        for idx in range(l_ang.getNumAngles()):
            a1, a2, a3, angle, k = l_ang.getAngleParameters(idx)
            p_ang.addAngle(lig_offset + a1, lig_offset + a2, lig_offset + a3, angle, k)

    # 3. PeriodicTorsionForce
    p_tor = get_force(prot_sys, openmm.PeriodicTorsionForce)
    l_tor = get_force(lig_sys, openmm.PeriodicTorsionForce)
    if l_tor is not None:
        for idx in range(l_tor.getNumTorsions()):
            a1, a2, a3, a4, per, phase, k = l_tor.getTorsionParameters(idx)
            p_tor.addTorsion(lig_offset + a1, lig_offset + a2, lig_offset + a3, lig_offset + a4, per, phase, k)

    # 4. NonbondedForce (both Amber14SB and OpenFF Sage use Lorentz-Berthelot + 1-4 scaling)
    p_nb = get_force(prot_sys, openmm.NonbondedForce)
    l_nb = get_force(lig_sys, openmm.NonbondedForce)
    p_nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for i in range(l_nb.getNumParticles()):
        q, sig, eps = l_nb.getParticleParameters(i)
        p_nb.addParticle(q, sig, eps)
    for j in range(l_nb.getNumExceptions()):
        a1, a2, qprod, sig, eps = l_nb.getExceptionParameters(j)
        p_nb.addException(lig_offset + a1, lig_offset + a2, qprod, sig, eps)

    # Build combined topology & positions
    lig_top = off_mol.to_topology().to_openmm()
    for res in lig_top.residues():
        res.name = "LIG"
    lig_pos = [openmm.Vec3(float(c[0]), float(c[1]), float(c[2])) * ounit.angstrom for c in lig_coords_ang]

    modeller = app.Modeller(prot_top, prot_pos)
    modeller.add(lig_top, lig_pos)

    z_all = [atom.element.atomic_number for atom in modeller.topology.atoms()]
    add_obc2_implicit_solvent(prot_sys, z_all)

    return modeller.topology, modeller.positions, prot_sys, lig_offset


def run_openff_pfba_torsion_and_md(off_pfba: OffMolecule, rd_pfba: Chem.Mol) -> Tuple[pd.DataFrame, List[float]]:
    """Runs OpenFF Sage 2.2.0 1D relaxed torsion scan AND 10 ps 300 K free-ligand MD on PFBA."""
    patt = Chem.MolFromSmarts("[CX3](=O)([O-])[CX4](F)(F)[CX4](F)(F)[CX4](F)(F)F")
    match = rd_pfba.GetSubstructMatch(patt)
    dihed_idx = (match[0], match[3], match[6], match[9])  # C1-Ca-Cb-Cg

    top = off_pfba.to_topology().to_openmm()
    conf = rd_pfba.GetConformer(0)
    init_pos = [
        openmm.Vec3(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z) * ounit.angstrom
        for i in range(rd_pfba.GetNumAtoms())
    ]

    # Part A: 1D relaxed torsion scan
    sys_scan = build_openff_ligand_system(off_pfba, include_obc2=True)
    torsion_restraint = openmm.CustomTorsionForce(
        "0.5 * k * min(dtheta, 2*pi - dtheta)^2; dtheta = abs(theta - theta0); pi = 3.141592653589793"
    )
    torsion_restraint.addGlobalParameter("k", 1000.0)
    torsion_restraint.addGlobalParameter("theta0", 0.0)
    torsion_restraint.addTorsion(*dihed_idx)
    torsion_restraint.setForceGroup(1)
    sys_scan.addForce(torsion_restraint)

    sim_scan = app.Simulation(
        top, sys_scan, openmm.VerletIntegrator(0.001 * ounit.picoseconds), openmm.Platform.getPlatformByName("CPU")
    )
    sim_scan.context.setPositions(init_pos)

    rows = []
    for deg in range(0, 361, 10):
        rad = math.radians(deg if deg <= 180 else deg - 360)
        sim_scan.context.setParameter("theta0", rad)
        sim_scan.minimizeEnergy(maxIterations=1000)
        e_kcal = sim_scan.context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(
            ounit.kilocalories_per_mole
        )
        rows.append({"angle_deg": float(deg), "energy_kcal": e_kcal})

    min_e = min(r["energy_kcal"] for r in rows)
    df_tor = pd.DataFrame([
        {"angle_deg": r["angle_deg"], "openff_delta_e_kcal_mol": r["energy_kcal"] - min_e}
        for r in rows
    ])

    # Part B: 10 ps (10,000 steps @ 1 fs) 300 K Langevin MD of free PFBA in OBC2
    sys_md = build_openff_ligand_system(off_pfba, include_obc2=True)
    integ_md = openmm.LangevinMiddleIntegrator(
        300.0 * ounit.kelvin, 2.0 / ounit.picoseconds, 0.001 * ounit.picoseconds
    )
    sim_md = app.Simulation(top, sys_md, integ_md, openmm.Platform.getPlatformByName("CPU"))
    sim_md.context.setPositions(init_pos)
    sim_md.minimizeEnergy(maxIterations=1000)
    sim_md.context.setVelocitiesToTemperature(300.0 * ounit.kelvin, 42)

    sampled_abs_diheds = []
    mol_tmp = Chem.Mol(rd_pfba)
    for _ in range(200):
        sim_md.step(50)
        pos_A = sim_md.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(ounit.angstrom)
        c_tmp = mol_tmp.GetConformer(0)
        for k in range(mol_tmp.GetNumAtoms()):
            c_tmp.SetAtomPosition(k, pos_A[k].tolist())
        phi = rdMolTransforms.GetDihedralDeg(c_tmp, *dihed_idx)
        sampled_abs_diheds.append(abs(float(phi)))

    return df_tor, sampled_abs_diheds


def extract_fah_template_from_cif(cif_path: str) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray]]:
    """Extracts Chain A FAH 305 heavy-atom coordinates, prochiral vectors, and protein CA map from 3R3V."""
    fah = {}
    ca_by_auth_seq = {}
    with open(cif_path) as f:
        headers, in_atom = [], False
        for line in f:
            s = line.strip()
            if s == "loop_":
                in_atom, headers = False, []
                continue
            if s.startswith("_atom_site."):
                in_atom = True
                headers.append(s.split(".")[1])
                continue
            if in_atom and s and not s.startswith("#") and not s.startswith("_"):
                parts = s.split()
                if len(parts) >= len(headers):
                    row = dict(zip(headers, parts))
                    if row["auth_asym_id"] == "A":
                        if row["auth_comp_id"] == "FAH":
                            fah[row["label_atom_id"]] = np.array([
                                float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])
                            ])
                        elif row["group_PDB"] == "ATOM" and row["label_atom_id"] == "CA":
                            ca_by_auth_seq[int(row["auth_seq_id"])] = np.array([
                                float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])
                            ])
    ca, c1, f_sc = fah["CH3"], fah["C"], fah["F"]
    u1 = (c1 - ca) / np.linalg.norm(c1 - ca)
    u2 = (f_sc - ca) / np.linalg.norm(f_sc - ca)
    bisector = -(u1 + u2) / np.linalg.norm(u1 + u2)
    perp = np.cross(u1, u2)
    perp = perp / np.linalg.norm(perp)
    half = math.radians(54.735)
    fah["v1"] = (bisector * math.cos(half) + perp * math.sin(half))
    fah["v2"] = (bisector * math.cos(half) - perp * math.sin(half))
    return fah, ca_by_auth_seq


def map_auth_residues_by_ca(top: app.Topology, pos: List[openmm.Vec3], ca_by_auth_seq: Dict[int, np.ndarray], target_seqs: List[int]) -> Dict[int, app.Residue]:
    """Maps crystallographic auth_seq_id integers to OpenMM Residue objects via exact CA proximity."""
    pos_A = np.array([[p.value_in_unit(ounit.angstrom)[0], p.value_in_unit(ounit.angstrom)[1], p.value_in_unit(ounit.angstrom)[2]] for p in pos])
    res_ca = {}
    for atom in top.atoms():
        if atom.name == "CA":
            res_ca[atom.residue.index] = (atom.residue, pos_A[atom.index])
    out = {}
    for seq_id in target_seqs:
        ref_pt = ca_by_auth_seq[seq_id]
        best_res = min(res_ca.values(), key=lambda item: float(np.linalg.norm(item[1] - ref_pt)))[0]
        out[seq_id] = best_res
    return out


def prepare_rpa1163_chain_a(
    cif_path: str,
    target_muts: Dict[int, str],
    ca_by_auth_seq: Dict[int, np.ndarray],
    out_pdb: str,
) -> Tuple[app.Topology, List[openmm.Vec3], Dict[int, int]]:
    """Prepares RPA1163 Chain A closing internal loop Gln255 while stripping terminal overhangs."""
    fixer = pdbfixer.PDBFixer(filename=cif_path)
    chains_to_remove = [i for i, ch in enumerate(fixer.topology.chains()) if ch.id != "A"]
    if chains_to_remove:
        fixer.removeChains(chains_to_remove)
    fixer.removeHeterogens(keepWater=False)

    # Map target mutation auth_seq_ids via CA coordinates so we never depend on mmCIF label vs auth id
    auth_map_pre = map_auth_residues_by_ca(fixer.topology, fixer.positions, ca_by_auth_seq, list(target_muts.keys()))
    mut_strings = [f"{auth_map_pre[seq].name}-{auth_map_pre[seq].id}-{new_aa}" for seq, new_aa in target_muts.items()]
    chain_id = list(fixer.topology.chains())[0].id
    if mut_strings:
        fixer.applyMutations(mut_strings, chain_id)

    fixer.findMissingResidues()
    # Keep internal missing residues (e.g., single disordered Gln255) to prevent broken backbone; strip N/C termini
    n_res_chain0 = sum(1 for _ in list(fixer.topology.chains())[0].residues())
    fixer.missingResidues = {
        k: v for k, v in fixer.missingResidues.items()
        if 0 < k[1] < n_res_chain0
    }
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    ff = app.ForceField("amber14-all.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)

    key_seqs = [110, 111, 114, 134, 155, 156, 219, 253, 280]
    auth_map_post = map_auth_residues_by_ca(modeller.topology, modeller.positions, ca_by_auth_seq, key_seqs)
    idx_155 = auth_map_post[155].index
    idx_280 = auth_map_post[280].index

    variants = []
    for res in modeller.topology.residues():
        if res.index == idx_155 and res.name == "HIS":
            variants.append("HIE")  # NE2-HE2 donates H-bond to scissile F
        elif res.index == idx_280 and res.name == "HIS":
            variants.append("HID")  # ND1-HD1 to Asp134; NE2 is general base
        else:
            variants.append(None)

    modeller.addHydrogens(ff, pH=7.4, variants=variants)
    with open(out_pdb, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    final_map = map_auth_residues_by_ca(modeller.topology, modeller.positions, ca_by_auth_seq, key_seqs)
    auth_to_res_index = {seq: r.index for seq, r in final_map.items()}
    return modeller.topology, modeller.positions, auth_to_res_index


def extract_zone0_heavy_coords_from_cif(cif_path: str) -> np.ndarray:
    """Extracts conserved Zone-0 heavy-atom coordinates from 3R3V Chain A for rotamer steric clearance."""
    zone0_seqs = {"110", "111", "114", "134", "155", "156", "219", "280"}
    pts = []
    with open(cif_path) as f:
        headers, in_atom = [], False
        for line in f:
            s = line.strip()
            if s == "loop_":
                in_atom, headers = False, []
                continue
            if s.startswith("_atom_site."):
                in_atom = True
                headers.append(s.split(".")[1])
                continue
            if in_atom and s and not s.startswith("#") and not s.startswith("_"):
                parts = s.split()
                if len(parts) >= len(headers):
                    row = dict(zip(headers, parts))
                    if (
                        row["group_PDB"] == "ATOM"
                        and row["auth_asym_id"] == "A"
                        and row["auth_seq_id"] in zone0_seqs
                        and row["type_symbol"] != "H"
                    ):
                        pts.append([float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])])
    return np.array(pts, dtype=float)


def align_substrate_to_fah(
    rd_mol: Chem.Mol,
    fah: Dict[str, np.ndarray],
    zone0_coords: np.ndarray,
) -> Tuple[Chem.Mol, Dict[str, int]]:
    """Stereoselectively aligns MFA, DFA, or PFBA onto FAH 305 and optimizes tail torsion away from Zone 0."""
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

    ca_atom = mol.GetAtomWithIdx(ca_idx)
    f_alpha_idxs = [n.GetIdx() for n in ca_atom.GetNeighbors() if n.GetSymbol() == "F"]
    cb_idxs = [n.GetIdx() for n in ca_atom.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != c1_idx]
    target_pts_base = [fah["C"], fah["O"], fah["OXT"], fah["CH3"], fah["F"]]

    best_mol, best_score, best_roles = None, -1e9, None
    for o_perm in [(o_idxs[0], o_idxs[1]), (o_idxs[1], o_idxs[0])]:
        for f_sc in f_alpha_idxs:
            base_trial = Chem.Mol(mol)
            prb_indices = [c1_idx, o_perm[0], o_perm[1], ca_idx, f_sc]
            ref_mol = Chem.RWMol()
            for _ in range(len(prb_indices)):
                ref_mol.AddAtom(Chem.Atom(6))
            ref_conf = Chem.Conformer(len(prb_indices))
            for k, pt in enumerate(target_pts_base):
                ref_conf.SetAtomPosition(k, pt.tolist())
            ref_mol.AddConformer(ref_conf)

            # Sweep C1-Ca-Cb-Cg torsion if present so PFBA enters in its extended pocket rotamer instead of 60 deg curl
            torsion_grid = range(0, 360, 15) if cb_idxs else [None]
            cg_idx = None
            if cb_idxs:
                cb_atom = base_trial.GetAtomWithIdx(cb_idxs[0])
                cg_candidates = [n.GetIdx() for n in cb_atom.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != ca_idx]
                cg_idx = cg_candidates[0] if cg_candidates else None

            for tor_deg in torsion_grid:
                trial = Chem.Mol(base_trial)
                if tor_deg is not None and cg_idx is not None:
                    rdMolTransforms.SetDihedralDeg(trial.GetConformer(0), c1_idx, ca_idx, cb_idxs[0], cg_idx, float(tor_deg))
                rmsd = rdMolAlign.AlignMol(trial, ref_mol, atomMap=[(prb_indices[k], k) for k in range(len(prb_indices))])
                t_conf = trial.GetConformer(0)
                ca_pos = np.array(t_conf.GetAtomPosition(ca_idx))
                score = -rmsd

                if cb_idxs:
                    cb_pos = np.array(t_conf.GetAtomPosition(cb_idxs[0]))
                    u_cb = (cb_pos - ca_pos) / np.linalg.norm(cb_pos - ca_pos)
                    score += 10.0 * float(np.dot(u_cb, fah["v2"]))
                    # Evaluate minimum clearance between distal tail atoms (Cb, Cg, tail Fs) and conserved Zone-0 core
                    tail_check_idxs = cb_idxs + ([cg_idx] if cg_idx is not None else []) + [
                        a.GetIdx() for a in trial.GetAtoms() if a.GetSymbol() == "F" and a.GetIdx() not in f_alpha_idxs
                    ]
                    tail_pts = np.array([t_conf.GetAtomPosition(ti) for ti in tail_check_idxs])
                    dmat = np.linalg.norm(tail_pts[:, None, :] - zone0_coords[None, :, :], axis=-1)
                    min_z0_dist = float(np.min(dmat))
                    score += 4.0 * min_z0_dist
                elif len(f_alpha_idxs) == 2:
                    f_other = [x for x in f_alpha_idxs if x != f_sc][0]
                    fo_pos = np.array(t_conf.GetAtomPosition(f_other))
                    u_fo = (fo_pos - ca_pos) / np.linalg.norm(fo_pos - ca_pos)
                    score += 5.0 * float(np.dot(u_fo, fah["v2"]))

                if score > best_score:
                    best_score = score
                    best_mol = trial
                    tail_f_idxs = [a.GetIdx() for a in trial.GetAtoms() if a.GetSymbol() == "F" and a.GetIdx() not in f_alpha_idxs]
                    best_roles = {
                        "c1": c1_idx,
                        "o1": o_perm[0],
                        "o2": o_perm[1],
                        "ca": ca_idx,
                        "f_scissile": f_sc,
                        "f_alpha_all": f_alpha_idxs,
                        "tail_f_all": tail_f_idxs,
                    }
    return best_mol, best_roles


def simulate_and_score_holo(
    prot_top: app.Topology,
    prot_pos: List[openmm.Vec3],
    auth_to_res_index: Dict[int, int],
    off_mol: OffMolecule,
    placed_rd_mol: Chem.Mol,
    roles: Dict[str, int],
    fah_anchor_ca: np.ndarray,
    out_min_pdb: str,
    steps: int = 4000,
    report_interval: int = 80,
) -> Dict:
    """Combines receptor + OpenFF Sage ligand, minimizes, runs 300 K active-site dynamics, and scores NAC + E_pull."""
    conf = placed_rd_mol.GetConformer(0)
    coords_ang = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(placed_rd_mol.GetNumAtoms())
    ])

    top_all, pos_all, system, lig_offset = merge_protein_and_openff_ligand_systems(
        prot_top, prot_pos, off_mol, coords_ang
    )

    # Restrain outer-shell C-alpha atoms (> 10.0 A from pocket center) so entire pocket & cap breathe freely
    restraint = openmm.CustomExternalForce("0.5 * k_pos * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
    restraint.addGlobalParameter("k_pos", 418.4)  # 1.0 kcal/(mol*A^2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    pos_nm = [p.value_in_unit(ounit.nanometer) for p in pos_all]
    for atom in top_all.atoms():
        if atom.residue.name != "LIG" and atom.name == "CA":
            p_ang = np.array([pos_nm[atom.index][0], pos_nm[atom.index][1], pos_nm[atom.index][2]]) * 10.0
            if np.linalg.norm(p_ang - fah_anchor_ca) > 10.0:
                restraint.addParticle(atom.index, pos_nm[atom.index])
    system.addForce(restraint)

    nb_force = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    prot_atom_indices = np.arange(0, lig_offset, dtype=int)
    prot_charges_e = np.array(
        [nb_force.getParticleParameters(int(i))[0].value_in_unit(ounit.elementary_charge) for i in prot_atom_indices],
        dtype=float,
    )

    r110 = auth_to_res_index[110]
    r111 = auth_to_res_index[111]
    r114 = auth_to_res_index[114]
    r155 = auth_to_res_index[155]
    r156 = auth_to_res_index[156]
    r219 = auth_to_res_index[219]

    idx_asp110_od, idx_cradle_H, idx_cradle_heavy, idx_clamp_N = [], [], [], []
    for atom in top_all.atoms():
        if atom.index >= lig_offset:
            continue
        ridx, aname = atom.residue.index, atom.name
        if ridx == r110 and aname in ("OD1", "OD2"):
            idx_asp110_od.append(atom.index)
        if (ridx == r155 and aname == "HE2") or (ridx == r156 and aname == "HE1") or (ridx == r219 and aname == "HH"):
            idx_cradle_H.append(atom.index)
        if (ridx == r155 and aname == "NE2") or (ridx == r156 and aname == "NE1") or (ridx == r219 and aname == "OH"):
            idx_cradle_heavy.append(atom.index)
        if ridx in (r111, r114) and aname in ("NE", "NH1", "NH2"):
            idx_clamp_N.append(atom.index)

    idx_ca = lig_offset + roles["ca"]
    idx_o1 = lig_offset + roles["o1"]
    idx_o2 = lig_offset + roles["o2"]
    idx_f_alpha_all = [lig_offset + i for i in roles["f_alpha_all"]]
    idx_tail_f_all = [lig_offset + i for i in roles["tail_f_all"]]

    # 1.0 fs step (unconstrained bonds) for crisp numerical stability
    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * ounit.kelvin, 2.0 / ounit.picoseconds, 0.001 * ounit.picoseconds
    )
    sim = app.Simulation(top_all, system, integrator)
    sim.context.setPositions(pos_all)

    e_init = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(ounit.kilocalories_per_mole)
    sim.minimizeEnergy(maxIterations=1500)
    min_state = sim.context.getState(getEnergy=True, getPositions=True)
    e_min = min_state.getPotentialEnergy().value_in_unit(ounit.kilocalories_per_mole)

    os.makedirs(os.path.dirname(out_min_pdb), exist_ok=True)
    with open(out_min_pdb, "w") as f:
        app.PDBFile.writeFile(top_all, min_state.getPositions(), f, keepIds=True)

    min_pos_A = min_state.getPositions(asNumpy=True).value_in_unit(ounit.angstrom)
    ca_shift_min_A = float(np.linalg.norm(min_pos_A[idx_ca] - fah_anchor_ca))

    sim.context.setVelocitiesToTemperature(300.0 * ounit.kelvin, 42)
    n_frames = steps // report_interval

    d_attacks, ang_attacks, cradle_counts, clamp_dists, e_pulls = [], [], [], [], []
    nac_strict_flags, nac_relaxed_flags, inverted_flags, ca_positions = [], [], [], []

    for _ in range(n_frames):
        sim.step(report_interval)
        pos_A = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(ounit.angstrom)
        r_ca = pos_A[idx_ca]
        ca_positions.append(r_ca.copy())

        d_clamp = min(
            float(np.linalg.norm(pos_A[idx_n] - pos_A[idx_o]))
            for idx_n in idx_clamp_N
            for idx_o in (idx_o1, idx_o2)
        )

        best_f_tuple = None
        for idx_f in idx_f_alpha_all:
            r_f = pos_A[idx_f]
            n_hb = sum(1 for ih in idx_cradle_H if np.linalg.norm(pos_A[ih] - r_f) <= 2.50)
            d_od_list = [(float(np.linalg.norm(pos_A[iod] - r_ca)), iod) for iod in idx_asp110_od]
            d_od_min, iod_best = min(d_od_list, key=lambda x: x[0])
            u_od = (pos_A[iod_best] - r_ca) / np.linalg.norm(pos_A[iod_best] - r_ca)
            u_f = (r_f - r_ca) / np.linalg.norm(r_f - r_ca)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u_od, u_f))))))

            is_nac_strict = (d_od_min <= 3.30) and (155.0 <= ang <= 180.0) and (n_hb >= 2) and (d_clamp <= 3.40)
            is_nac_relax = (d_od_min <= 3.50) and (140.0 <= ang <= 180.0) and (n_hb >= 2) and (d_clamp <= 3.50)
            rank_key = (int(is_nac_strict), int(is_nac_relax), n_hb, ang, -d_od_min)
            if best_f_tuple is None or rank_key > best_f_tuple[0]:
                best_f_tuple = (rank_key, idx_f, d_od_min, ang, n_hb, is_nac_strict, is_nac_relax)

        _, idx_f_sc, d_att, ang_att, n_hb, nac_s, nac_r = best_f_tuple
        r_f_sc = pos_A[idx_f_sc]

        r_mid = 0.5 * (r_ca + r_f_sc)
        u_cf = (r_f_sc - r_ca) / np.linalg.norm(r_f_sc - r_ca)
        diff = r_mid[None, :] - pos_A[prot_atom_indices]
        r_norm = np.linalg.norm(diff, axis=1)
        e_vec = KE_MV_CM_PER_E_ANG2 * np.sum((prot_charges_e[:, None] * diff) / (r_norm[:, None] ** 3), axis=0)
        e_pull = float(-np.dot(e_vec, u_cf))

        cradle_center = np.mean(pos_A[idx_cradle_heavy], axis=0)
        d_alpha_cradle = min(float(np.linalg.norm(pos_A[i] - cradle_center)) for i in idx_f_alpha_all)
        if idx_tail_f_all:
            d_tail_cradle = min(float(np.linalg.norm(pos_A[i] - cradle_center)) for i in idx_tail_f_all)
            inverted = d_tail_cradle < d_alpha_cradle
        else:
            inverted = False

        d_attacks.append(d_att)
        ang_attacks.append(ang_att)
        cradle_counts.append(n_hb)
        clamp_dists.append(d_clamp)
        e_pulls.append(e_pull)
        nac_strict_flags.append(float(nac_s))
        nac_relaxed_flags.append(float(nac_r))
        inverted_flags.append(float(inverted))

    ca_arr = np.array(ca_positions)
    ca_rmsf = float(np.sqrt(np.mean(np.sum((ca_arr - np.mean(ca_arr, axis=0)) ** 2, axis=1))))

    return {
        "e_init_kcal_mol": round(e_init, 1),
        "e_min_kcal_mol": round(e_min, 1),
        "ca_shift_after_min_A": round(ca_shift_min_A, 3),
        "mean_d_attack_A": round(float(np.mean(d_attacks)), 3),
        "min_d_attack_A": round(float(np.min(d_attacks)), 3),
        "mean_ang_SN2_deg": round(float(np.mean(ang_attacks)), 1),
        "max_ang_SN2_deg": round(float(np.max(ang_attacks)), 1),
        "mean_cradle_hbonds": round(float(np.mean(cradle_counts)), 2),
        "mean_clamp_dist_A": round(float(np.mean(clamp_dists)), 3),
        "p_nac_strict_pct": round(100.0 * float(np.mean(nac_strict_flags)), 1),
        "p_nac_near_pct": round(100.0 * float(np.mean(nac_relaxed_flags)), 1),
        "mean_E_pull_MV_cm": round(float(np.mean(e_pulls)), 1),
        "ca_rmsf_A": round(ca_rmsf, 3),
        "inverted_pose_pct": round(100.0 * float(np.mean(inverted_flags)), 1),
        "minimized_pdb": out_min_pdb,
    }


def main():
    base_dir = "/Users/o2satz/antigravity/pfas"
    cif_3r3v = "/Users/o2satz/.gemini/antigravity/brain/1e66644c-6bfc-43f0-b691-dcbc01afc00e/structures/pdb_00003r3v.cif"
    sdf_ligs = os.path.join(base_dir, "data/ligands/substrates_ph74.sdf")
    out_dir = os.path.join(base_dir, "results/smoke_test")
    os.makedirs(out_dir, exist_ok=True)

    suppl = Chem.SDMolSupplier(sdf_ligs, removeHs=False)
    ligs_rd, ligs_off, charge_methods = {}, {}, {}
    for mol in suppl:
        if mol is None:
            continue
        cid = mol.GetProp("_Name").strip() if mol.HasProp("_Name") and mol.GetProp("_Name").strip() else mol.GetProp("compound_id")
        ligs_rd[cid] = mol
        off_m, q_method = assign_charges_openff(mol, cid)
        ligs_off[cid] = off_m
        charge_methods[cid] = q_method
        print(f"[OpenFF 2.2.0] Parameterized {cid}: charge_method={q_method}, net_q={sum([float(x.m) for x in off_m.partial_charges]):.3f} e")

    print("[Torsion & Free MD] Running OpenFF Sage 2.2.0 relaxed scan + 10 ps 300 K MD on PFBA...")
    df_openff_tor, md_abs_diheds = run_openff_pfba_torsion_and_md(ligs_off["PFBA"], ligs_rd["PFBA"])
    openff_tor_csv = os.path.join(out_dir, "pfba_openff_torsion.csv")
    df_openff_tor.to_csv(openff_tor_csv, index=False)
    print(
        f"   -> OpenFF PFBA 300K |C1-Ca-Cb-Cg| dihedral: mean={np.mean(md_abs_diheds):.1f} +/- {np.std(md_abs_diheds):.1f} deg "
        f"(gauche <90 deg fraction: {100.0 * np.mean(np.array(md_abs_diheds) < 90.0):.1f}%)"
    )

    # Also scan pure internal perfluoroalkyl -CF2-CF2-CF2-CF2- rotor (n-C4F10) in OpenFF Sage 2.2.0 to verify 15/7 helix!
    mol_c4f10 = Chem.AddHs(Chem.MolFromSmiles("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F"))
    AllChem.EmbedMolecule(mol_c4f10, AllChem.ETKDGv3())
    off_c4f10, _ = assign_charges_openff(mol_c4f10, "C4F10")
    c_idxs = [a.GetIdx() for a in mol_c4f10.GetAtoms() if a.GetSymbol() == "C"]
    sys_c4 = build_openff_ligand_system(off_c4f10, include_obc2=False)
    tor_c4 = openmm.CustomTorsionForce(
        "0.5 * k * min(dtheta, 2*pi - dtheta)^2; dtheta = abs(theta - theta0); pi = 3.141592653589793"
    )
    tor_c4.addGlobalParameter("k", 1000.0)
    tor_c4.addGlobalParameter("theta0", 0.0)
    tor_c4.addTorsion(*c_idxs)
    tor_c4.setForceGroup(1)
    sys_c4.addForce(tor_c4)
    sim_c4 = app.Simulation(
        off_c4f10.to_topology().to_openmm(),
        sys_c4,
        openmm.VerletIntegrator(0.001 * ounit.picoseconds),
        openmm.Platform.getPlatformByName("CPU"),
    )
    c4_conf = mol_c4f10.GetConformer(0)
    sim_c4.context.setPositions([
        openmm.Vec3(c4_conf.GetAtomPosition(i).x, c4_conf.GetAtomPosition(i).y, c4_conf.GetAtomPosition(i).z) * ounit.angstrom
        for i in range(mol_c4f10.GetNumAtoms())
    ])
    c4_rows = []
    for deg in range(0, 361, 10):
        rad = math.radians(deg if deg <= 180 else deg - 360)
        sim_c4.context.setParameter("theta0", rad)
        sim_c4.minimizeEnergy(maxIterations=1000)
        ec4 = sim_c4.context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(ounit.kilocalories_per_mole)
        c4_rows.append((float(deg), ec4))
    min_ec4 = min(x[1] for x in c4_rows)
    df_c4_tor = pd.DataFrame([{"angle_deg": d, "c4f10_delta_e_kcal_mol": e - min_ec4} for d, e in c4_rows])

    fah, ca_by_auth_seq = extract_fah_template_from_cif(cif_3r3v)
    zone0_coords = extract_zone0_heavy_coords_from_cif(cif_3r3v)
    receptors = {}
    for var_name, muts in [("WT", {110: "ASP"}), ("I253A", {110: "ASP", 253: "ALA"})]:
        rec_pdb = os.path.join(out_dir, f"receptor_{var_name}_ph74.pdb")
        print(f"[Receptor] Preparing {var_name} ({muts})...")
        receptors[var_name] = prepare_rpa1163_chain_a(cif_3r3v, muts, ca_by_auth_seq, rec_pdb)

    summary_rows = []
    for sub_id in ["MFA", "DFA", "PFBA"]:
        placed_mol, roles = align_substrate_to_fah(ligs_rd[sub_id], fah, zone0_coords)
        for var_name in ["WT", "I253A"]:
            print(f"[Holo MD] Running {var_name} + {sub_id} (minimize + 4 ps @ 300 K)...")
            prot_top, prot_pos, auth_to_res_index = receptors[var_name]
            out_pdb = os.path.join(out_dir, f"holo_{var_name}_{sub_id}_minimized.pdb")
            metrics = simulate_and_score_holo(
                prot_top=prot_top,
                prot_pos=prot_pos,
                auth_to_res_index=auth_to_res_index,
                off_mol=ligs_off[sub_id],
                placed_rd_mol=placed_mol,
                roles=roles,
                fah_anchor_ca=fah["CH3"],
                out_min_pdb=out_pdb,
                steps=4000,
                report_interval=80,
            )
            row = {"variant": var_name, "substrate": sub_id, "charge_method": charge_methods[sub_id], **metrics}
            summary_rows.append(row)
            print(
                f"   -> E_init={metrics['e_init_kcal_mol']:.0f} -> E_min={metrics['e_min_kcal_mol']:.0f} kcal/mol | "
                f"d_att={metrics['mean_d_attack_A']:.2f}A (min {metrics['min_d_attack_A']:.2f}A) | "
                f"ang={metrics['mean_ang_SN2_deg']:.1f}deg (max {metrics['max_ang_SN2_deg']:.1f}deg) | "
                f"cradle_HB={metrics['mean_cradle_hbonds']:.2f} | E_pull={metrics['mean_E_pull_MV_cm']:.1f} MV/cm | "
                f"Ca_shift={metrics['ca_shift_after_min_A']:.2f}A | inv={metrics['inverted_pose_pct']:.0f}%"
            )

    df_res = pd.DataFrame(summary_rows)
    res_csv = os.path.join(out_dir, "holo_smoke_test_metrics.csv")
    df_res.to_csv(res_csv, index=False)

    # Plot 1: OpenFF Sage 2.2.0 vs MMFF94s Torsion Profile
    set_safe_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained", dpi=300)
    ax.plot(
        df_openff_tor["angle_deg"],
        df_openff_tor["openff_delta_e_kcal_mol"],
        color=PUBLICATION_CYCLE[0],
        linewidth=2.4,
        label="PFBA Headgroup Rotor (-CO2⁻–CF2–CF2–CF3, OpenFF 2.2 + OBC2)",
    )
    ax.plot(
        df_c4_tor["angle_deg"],
        df_c4_tor["c4f10_delta_e_kcal_mol"],
        color=PUBLICATION_CYCLE[2],
        linewidth=2.4,
        label="Internal Perfluoro Chain (CF3–CF2–CF2–CF3, OpenFF 2.2)",
    )
    mmff_csv = os.path.join(out_dir, "pfba_mmff94s_torsion.csv")
    if os.path.exists(mmff_csv):
        df_mmff = pd.read_csv(mmff_csv)
        ax.plot(
            df_mmff["angle_deg"],
            df_mmff["delta_e_kcal_mol"],
            color=PUBLICATION_CYCLE[1],
            linestyle="--",
            linewidth=1.9,
            label="PFBA Headgroup Rotor (MMFF94s gas-phase artifact)",
        )
    ax.axvspan(155, 205, color=PUBLICATION_CYCLE[2], alpha=0.12, label="15/7 Perfluoro Helix Basin (~163° / 197°)")
    ax.set_xlabel("Backbone C–C–C–C Dihedral Angle (degrees)")
    ax.set_ylabel("Relative Energy ΔE (kcal/mol)")
    ax.set_title("Fluorocarbon Torsion Physics: Internal Helix vs. Headgroup Curl")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.legend(frameon=True, fontsize=8)
    apply_publication_spines(ax)
    tor_plot_path = os.path.join(out_dir, "pfba_openff_vs_mmff_torsion.png")
    fig.savefig(tor_plot_path)
    plt.close(fig)

    # Plot 2: WT vs I253A Active-Site Pre-Organization Comparison
    set_safe_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), layout="constrained", dpi=300)
    subs = ["MFA", "DFA", "PFBA"]
    x = np.arange(len(subs))
    w = 0.35

    wt_rows = [df_res[(df_res["variant"] == "WT") & (df_res["substrate"] == s)].iloc[0] for s in subs]
    mut_rows = [df_res[(df_res["variant"] == "I253A") & (df_res["substrate"] == s)].iloc[0] for s in subs]

    axes[0].bar(x - w/2, [r["mean_d_attack_A"] for r in wt_rows], w, label="WT (Ile253)", color=PUBLICATION_CYCLE[0])
    axes[0].bar(x + w/2, [r["mean_d_attack_A"] for r in mut_rows], w, label="I253A (Alcove)", color=PUBLICATION_CYCLE[2])
    axes[0].axhline(3.30, color="#DC2626", linestyle="--", linewidth=1.2, label="NAC Cutoff (3.3 Å)")
    axes[0].set_xticks(x, subs)
    axes[0].set_ylabel("Mean d(Asp110-Oδ ··· Cα) (Å)")
    axes[0].set_title("Nucleophile Attack Distance")
    axes[0].legend(fontsize=8)
    apply_publication_spines(axes[0])

    axes[1].bar(x - w/2, [r["mean_ang_SN2_deg"] for r in wt_rows], w, label="WT", color=PUBLICATION_CYCLE[0])
    axes[1].bar(x + w/2, [r["mean_ang_SN2_deg"] for r in mut_rows], w, label="I253A", color=PUBLICATION_CYCLE[2])
    axes[1].axhline(155.0, color="#DC2626", linestyle="--", linewidth=1.2, label="Strict NAC (155°)")
    axes[1].set_xticks(x, subs)
    axes[1].set_ylabel("Mean θ(Asp110-Oδ – Cα – F) (°)")
    axes[1].set_title("Backside SN2 Colinearity")
    axes[1].set_ylim(75, 185)
    axes[1].legend(fontsize=8)
    apply_publication_spines(axes[1])

    axes[2].bar(x - w/2, [r["mean_E_pull_MV_cm"] for r in wt_rows], w, label="WT", color=PUBLICATION_CYCLE[0])
    axes[2].bar(x + w/2, [r["mean_E_pull_MV_cm"] for r in mut_rows], w, label="I253A", color=PUBLICATION_CYCLE[2])
    axes[2].axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)
    axes[2].set_xticks(x, subs)
    axes[2].set_ylabel("Mean Scissile E_pull (MV/cm)")
    axes[2].set_title("Cα–F Polarizing Electric Field")
    axes[2].legend(fontsize=8)
    apply_publication_spines(axes[2])

    holo_plot_path = os.path.join(out_dir, "wt_vs_i253a_smoke_summary.png")
    fig.savefig(holo_plot_path)
    plt.close(fig)

    summary_json = os.path.join(out_dir, "smoke_test_summary.json")
    with open(summary_json, "w") as f:
        json.dump(
            {
                "openff_pfba_min_angle_deg": float(df_openff_tor.loc[df_openff_tor["openff_delta_e_kcal_mol"].idxmin(), "angle_deg"]),
                "openff_pfba_barrier_kcal_mol": round(float(df_openff_tor["openff_delta_e_kcal_mol"].max()), 2),
                "openff_pfba_300K_mean_abs_dihed_deg": round(float(np.mean(md_abs_diheds)), 1),
                "openff_pfba_300K_std_abs_dihed_deg": round(float(np.std(md_abs_diheds)), 1),
                "openff_pfba_300K_gauche_pct": round(float(100.0 * np.mean(np.array(md_abs_diheds) < 90.0)), 1),
                "holo_results": summary_rows,
            },
            f,
            indent=2,
        )
    print(f"\n[Done] Wrote {res_csv}, {tor_plot_path}, and {holo_plot_path}")


if __name__ == "__main__":
    main()
