"""Receptor preparation, coordinate-anchored mutagenesis, loop closure, and AF2 fold gate."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gemmi
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as ounit
import pdbfixer

from pfas_bench.config import RubricSpec, VariantSpec


@dataclass
class CrystalReference3R3V:
    fah_atoms: Dict[str, np.ndarray]
    v1_cleft: np.ndarray
    v2_alcove: np.ndarray
    ca_by_chain_seq: Dict[Tuple[str, int], np.ndarray]
    zone0_heavy_coords_A: np.ndarray
    zone0_residue_heavy_map_A: Dict[Tuple[int, str], np.ndarray]


@dataclass
class PreparedReceptor:
    variant: VariantSpec
    assembly: str
    ph: float
    topology: app.Topology
    positions: List[openmm.Vec3]
    catalytic_res_indices: Dict[int, int]  # auth_seq_id (Chain A) -> OpenMM residue.index


def extract_3r3v_reference(
    cif_path: str | Path,
    zone0_seqs: Sequence[int] = (110, 111, 114, 134, 155, 156, 219, 280),
) -> CrystalReference3R3V:
    """Extracts Chain A FAH 305 anchor, tetrahedral vectors, CA map (chains A & B), and Zone-0 coords."""
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()
    st = gemmi.make_structure_from_block(block)
    model = st[0]

    fah_atoms: Dict[str, np.ndarray] = {}
    ca_by_chain_seq: Dict[Tuple[str, int], np.ndarray] = {}
    z0_pts: List[np.ndarray] = []
    z0_map: Dict[Tuple[int, str], np.ndarray] = {}
    z0_set = set(int(x) for x in zone0_seqs)

    for chain in model:
        cname = chain.name
        for res in chain:
            seq_num = res.seqid.num
            if cname == "A" and res.name == "FAH":
                for atom in res:
                    fah_atoms[atom.name] = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            elif res.entity_type == gemmi.EntityType.Polymer:
                for atom in res:
                    if atom.is_hydrogen():
                        continue
                    pt = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                    if atom.name == "CA":
                        ca_by_chain_seq[(cname, seq_num)] = pt
                    if cname == "A" and seq_num in z0_set:
                        z0_pts.append(pt)
                        # Skip residue 110 sidechain OD/ND ambiguity for WT vs D110N RMSD comparison
                        if not (seq_num == 110 and atom.name in ("OD1", "OD2", "ND2")):
                            z0_map[(seq_num, atom.name)] = pt

    ca = fah_atoms["CH3"]
    c1 = fah_atoms["C"]
    f_sc = fah_atoms["F"]
    u1 = (c1 - ca) / np.linalg.norm(c1 - ca)
    u2 = (f_sc - ca) / np.linalg.norm(f_sc - ca)
    bisector = -(u1 + u2) / np.linalg.norm(u1 + u2)
    perp = np.cross(u1, u2)
    perp = perp / np.linalg.norm(perp)
    half = math.radians(54.735)
    v1 = bisector * math.cos(half) + perp * math.sin(half)
    v2 = bisector * math.cos(half) - perp * math.sin(half)

    return CrystalReference3R3V(
        fah_atoms=fah_atoms,
        v1_cleft=v1 / np.linalg.norm(v1),
        v2_alcove=v2 / np.linalg.norm(v2),
        ca_by_chain_seq=ca_by_chain_seq,
        zone0_heavy_coords_A=np.array(z0_pts, dtype=float),
        zone0_residue_heavy_map_A=z0_map,
    )


def _match_residues_by_ca(
    top: app.Topology,
    pos: Sequence[openmm.Vec3],
    ref: CrystalReference3R3V,
    chain_label: str,
    target_seqs: Sequence[int],
) -> Dict[int, app.Residue]:
    """Maps crystallographic (chain_label, auth_seq_id) to OpenMM Residue objects via CA proximity."""
    pos_A = np.array(
        [[p.value_in_unit(ounit.angstrom)[0], p.value_in_unit(ounit.angstrom)[1], p.value_in_unit(ounit.angstrom)[2]] for p in pos]
    )
    ca_candidates = []
    for atom in top.atoms():
        if atom.name == "CA":
            ca_candidates.append((atom.residue, pos_A[atom.index]))

    matched: Dict[int, app.Residue] = {}
    for seq_id in target_seqs:
        ref_pt = ref.ca_by_chain_seq[(chain_label, int(seq_id))]
        best_res, best_pt = min(ca_candidates, key=lambda item: float(np.linalg.norm(item[1] - ref_pt)))
        if float(np.linalg.norm(best_pt - ref_pt)) > 2.5:
            raise RuntimeError(f"Could not match {chain_label}:{seq_id} CA within 2.5 A in topology.")
        matched[int(seq_id)] = best_res
    return matched


def build_receptor_variant(
    cif_path: str | Path,
    ref: CrystalReference3R3V,
    variant: VariantSpec,
    assembly: str = "chain_A_monomer",
    ph: float = 7.5,
    out_pdb: str | Path | None = None,
) -> PreparedReceptor:
    """Prepares Chain A monomer or asymmetric AB homodimer with WT N110D reversion + variant mutations."""
    fixer = pdbfixer.PDBFixer(filename=str(cif_path))
    if assembly == "chain_A_monomer":
        keep_chains = {"A"}
    elif assembly == "asymmetric_homodimer_AB":
        keep_chains = {"A", "B"}
    else:
        raise ValueError(f"Unsupported assembly mode: {assembly}")

    remove_indices = [i for i, ch in enumerate(fixer.topology.chains()) if ch.id not in keep_chains]
    if remove_indices:
        fixer.removeChains(remove_indices)
    # Retain crystallographic buried active-site waters (e.g. catalytic water near His280 in 3R3V);
    # downstream build_holo_openmm_system automatically carves any water within 2.55 A of placed substrate!
    fixer.removeHeterogens(keepWater=True)

    # Compose effective residue-level mutations relative to 3R3V (which starts as D110N!)
    # Every variant EXCEPT D110N reverts 110 -> ASP, then overlays variant.mutations
    eff_muts: Dict[int, str] = {110: "ASP"}
    eff_muts.update(variant.mutations)
    if eff_muts.get(110) == "ASN":
        # 3R3V already has ASN110; remove no-op mutation so PDBFixer does not complain
        eff_muts.pop(110)

    if eff_muts:
        for ch in list(fixer.topology.chains()):
            cname = ch.id
            matched = _match_residues_by_ca(fixer.topology, fixer.positions, ref, cname, list(eff_muts.keys()))
            mut_strs = [
                f"{matched[seq].name}-{matched[seq].id}-{target_aa}"
                for seq, target_aa in eff_muts.items()
                if matched[seq].name != target_aa
            ]
            if mut_strs:
                fixer.applyMutations(mut_strs, cname)

    fixer.findMissingResidues()
    # Close internal missing loops (e.g. disordered Gln255 in 3R3V) while stripping N/C-terminal overhangs
    chain_lengths = {idx: sum(1 for _ in ch.residues()) for idx, ch in enumerate(fixer.topology.chains())}
    fixer.missingResidues = {
        (c_idx, r_idx): res_list
        for (c_idx, r_idx), res_list in fixer.missingResidues.items()
        if 0 < r_idx < chain_lengths.get(c_idx, 0)
    }
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)

    # Assign mechanistically accurate catalytic His states across active chains:
    # pH >= 6.0: His155=HIE (NE2-HE2 halide donor), His280=HID (ND1-HD1 to Asp134, NE2 base)
    # pH < 6.0 (acidic soil matrix stress): His280 protonates to HIP (+1), testing acid sensitivity
    his_155_indices = set()
    his_280_indices = set()
    for cname in sorted(keep_chains):
        m_his = _match_residues_by_ca(modeller.topology, modeller.positions, ref, cname, [155, 280])
        his_155_indices.add(m_his[155].index)
        his_280_indices.add(m_his[280].index)

    variants_list = []
    for res in modeller.topology.residues():
        if res.index in his_155_indices and res.name == "HIS":
            variants_list.append("HIE" if ph >= 5.0 else "HIP")
        elif res.index in his_280_indices and res.name == "HIS":
            variants_list.append("HID" if ph >= 6.0 else "HIP")
        else:
            variants_list.append(None)

    modeller.addHydrogens(ff, pH=ph, variants=variants_list)

    if out_pdb is not None:
        out_p = Path(out_pdb)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fh:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)

    key_seqs = [110, 111, 114, 134, 155, 156, 219, 253, 280]
    cat_map = _match_residues_by_ca(modeller.topology, modeller.positions, ref, "A", key_seqs)
    cat_indices = {seq: res.index for seq, res in cat_map.items()}

    return PreparedReceptor(
        variant=variant,
        assembly=assembly,
        ph=ph,
        topology=modeller.topology,
        positions=modeller.positions,
        catalytic_res_indices=cat_indices,
    )


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Computes optimal rigid-body superimposed RMSD between matched (N, 3) point sets."""
    Pc = P - np.mean(P, axis=0)
    Qc = Q - np.mean(Q, axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])
    R = Vt.T @ S @ U.T
    diff = (Pc @ R) - Qc
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def evaluate_fold_sanity_gate(
    predicted_structure_path: str | Path,
    ref: CrystalReference3R3V,
    rubric: RubricSpec,
    active_site_seqs: Sequence[int] = (40, 110, 111, 114, 134, 135, 141, 153, 155, 156, 181, 185, 219, 253, 280),
    is_experimental_or_homology: bool = False,
) -> Dict[str, float | bool]:
    """Evaluates Stage A Gate 1 (active-site pLDDT and Zone-0 heavy-atom RMSD vs 3R3V)."""
    st = gemmi.read_structure(str(predicted_structure_path))
    chain = st[0][0]

    plddt_vals: List[float] = []
    P_pts: List[np.ndarray] = []
    Q_pts: List[np.ndarray] = []
    active_set = set(int(s) for s in active_site_seqs)

    for res in chain:
        if res.entity_type != gemmi.EntityType.Polymer:
            continue
        snum = res.seqid.num
        for atom in res:
            if atom.is_hydrogen():
                continue
            if snum in active_set and atom.name == "CA":
                plddt_vals.append(float(atom.b_iso))
            key = (snum, atom.name)
            if key in ref.zone0_residue_heavy_map_A:
                P_pts.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float))
                Q_pts.append(ref.zone0_residue_heavy_map_A[key])

    zone0_rmsd = _kabsch_rmsd(np.array(P_pts), np.array(Q_pts)) if len(P_pts) >= 4 else 99.9
    if is_experimental_or_homology:
        # Do NOT fabricate pLDDT=100.0 for PDBFixer crystal-template point-mutant models; report provenance explicitly
        return {
            "structure_source": "crystal_3R3V_point_mutant",
            "active_site_plddt": None,
            "zone0_heavy_rmsd_A": round(zone0_rmsd, 3),
            "stage_a_fold_pass": bool(zone0_rmsd <= rubric.max_zone0_heavy_rmsd_A),
        }

    mean_plddt = float(np.mean(plddt_vals)) if plddt_vals else 0.0
    passed = (mean_plddt >= rubric.min_active_site_plddt) and (zone0_rmsd <= rubric.max_zone0_heavy_rmsd_A)
    return {
        "structure_source": "predicted_af2_or_boltz",
        "active_site_plddt": round(mean_plddt, 2),
        "zone0_heavy_rmsd_A": round(zone0_rmsd, 3),
        "stage_a_fold_pass": bool(passed),
    }
