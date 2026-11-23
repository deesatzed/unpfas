"""Small-molecule PFAS preparation, NAGL/AM1-BCC charging, and torsion auditing."""

from dataclasses import dataclass
import math
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as ounit
from openff.toolkit.topology import Molecule as OffMolecule
from openff.toolkit.typing.engines.smirnoff import ForceField as OffForceField
from openff.units import unit as off_unit
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from pfas_bench.config import SubstrateSpec

_OBC2_PARAMS_BY_Z = {
    1: (0.120, 0.85),
    6: (0.170, 0.72),
    7: (0.155, 0.79),
    8: (0.150, 0.85),
    9: (0.147, 0.88),
    15: (0.185, 0.86),
    16: (0.180, 0.96),
    17: (0.175, 0.80),
}


@dataclass
class PreparedLigand:
    spec: SubstrateSpec
    rd_mol: Chem.Mol
    off_mol: OffMolecule
    charge_method: str
    net_charge_e: float


def attach_obc2_solvent(system: openmm.System, atomic_numbers: Sequence[int]) -> None:
    """Attaches a native OpenMM GBSAOBCForce matching NonbondedForce particle charges."""
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    gb = openmm.GBSAOBCForce()
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    for idx, z in enumerate(atomic_numbers):
        q, _, _ = nb.getParticleParameters(idx)
        rad_nm, scale = _OBC2_PARAMS_BY_Z.get(int(z), (0.170, 0.80))
        gb.addParticle(q, rad_nm * ounit.nanometer, scale)
    system.addForce(gb)


def _preorient_carbon_chain_extended(mol_h: Chem.Mol, target_dihed_deg: float = 170.0) -> None:
    """Pre-sets C-C-C-C torsions to extended helical (~170 deg) to avoid gas-phase MMFF94s 60 deg curl."""
    conf = mol_h.GetConformer(0)
    visited = set()
    for bond in mol_h.GetBonds():
        a2, a3 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a2.GetSymbol() != "C" or a3.GetSymbol() != "C":
            continue
        key = tuple(sorted((a2.GetIdx(), a3.GetIdx())))
        if key in visited:
            continue
        c1_cands = [n.GetIdx() for n in a2.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != a3.GetIdx()]
        c4_cands = [n.GetIdx() for n in a3.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != a2.GetIdx()]
        if c1_cands and c4_cands:
            visited.add(key)
            rdMolTransforms.SetDihedralDeg(
                conf, c1_cands[0], a2.GetIdx(), a3.GetIdx(), c4_cands[0], float(target_dihed_deg)
            )


def assign_openff_charges(
    rd_mol: Chem.Mol,
    name: str,
    priority: Sequence[str] = ("openff-gnn-am1bcc-0.1.0-rc.3.pt", "am1bcc", "mmff94s_fallback"),
) -> Tuple[OffMolecule, str]:
    """Assigns partial charges via OpenFF NAGL AM1-BCC or sqm AM1-BCC, with explicit fallback warning."""
    off_mol = OffMolecule.from_rdkit(rd_mol, allow_undefined_stereo=True)
    off_mol.name = name
    formal_q = float(Chem.GetFormalCharge(rd_mol))

    for method in priority:
        if method in ("mmff94s_fallback", "gasteiger_fallback"):
            break
        try:
            off_mol.assign_partial_charges(partial_charge_method=method)
            return off_mol, method
        except Exception:
            continue

    warnings.warn(
        f"[{name}] AM1-BCC / NAGL backend unavailable; falling back to formal-charge-normalized MMFF94s charges.",
        RuntimeWarning,
        stacklevel=2,
    )
    mp = AllChem.MMFFGetMoleculeProperties(rd_mol, mmffVariant="MMFF94s")
    if mp is not None:
        charges = np.array([mp.GetMMFFPartialCharge(i) for i in range(rd_mol.GetNumAtoms())], dtype=float)
        used = "mmff94s_fallback"
    else:
        AllChem.ComputeGasteigerCharges(rd_mol)
        charges = np.array([float(a.GetProp("_GasteigerCharge")) for a in rd_mol.GetAtoms()], dtype=float)
        charges = np.nan_to_num(charges, nan=0.0)
        used = "gasteiger_fallback"

    charges = charges + (formal_q - float(np.sum(charges))) / len(charges)
    off_mol.partial_charges = charges * off_unit.elementary_charge
    return off_mol, used


def prepare_anionic_substrate(
    spec: SubstrateSpec,
    charge_method_priority: Sequence[str] = ("openff-gnn-am1bcc-0.1.0-rc.3.pt", "am1bcc", "mmff94s_fallback"),
    forcefield_name: str = "openff-2.2.0.offxml",
    random_seed: int = 42,
) -> PreparedLigand:
    """Embeds, pre-extends, and relaxes an anionic PFAS substrate directly under OpenFF Sage + OBC2."""
    raw = Chem.MolFromSmiles(spec.smiles)
    if raw is None:
        raise ValueError(f"Invalid SMILES for {spec.id}: {spec.smiles}")
    if Chem.GetFormalCharge(raw) != spec.formal_charge:
        raise ValueError(
            f"Substrate {spec.id} formal charge {Chem.GetFormalCharge(raw)} != expected {spec.formal_charge}"
        )

    mol_h = Chem.AddHs(raw)
    mol_h.SetProp("_Name", spec.id)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        params.useRandomCoords = True
        AllChem.EmbedMolecule(mol_h, params)

    # Pre-set carbon chain to extended ~170 deg before OpenFF + OBC2 relaxation (avoids gas-phase MMFF94s curl!)
    _preorient_carbon_chain_extended(mol_h, target_dihed_deg=170.0)
    off_mol, q_method = assign_openff_charges(mol_h, spec.id, priority=charge_method_priority)

    # Relax bond lengths and valence angles in OpenFF Sage + OBC2 while lightly holding extended C-C-C-C torsion
    off_ff = OffForceField(forcefield_name)
    sys_lig = off_ff.create_openmm_system(off_mol.to_topology(), charge_from_molecules=[off_mol])
    attach_obc2_solvent(sys_lig, [a.atomic_number for a in off_mol.atoms])

    # Keep C-C-C-C dihedrals near 170 deg during initial seed preparation so bound placement starts extended
    c_atoms = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetSymbol() == "C"]
    if len(c_atoms) >= 4:
        tor_hold = openmm.CustomTorsionForce("0.5 * k * min(dt, 2*pi - dt)^2; dt = abs(theta - theta0); pi = 3.141592653589793")
        tor_hold.addGlobalParameter("k", 200.0)
        tor_hold.addGlobalParameter("theta0", math.radians(170.0))
        for bond in mol_h.GetBonds():
            a2, a3 = bond.GetBeginAtom(), bond.GetEndAtom()
            if a2.GetSymbol() == "C" and a3.GetSymbol() == "C":
                c1s = [n.GetIdx() for n in a2.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != a3.GetIdx()]
                c4s = [n.GetIdx() for n in a3.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() != a2.GetIdx()]
                if c1s and c4s:
                    tor_hold.addTorsion(c1s[0], a2.GetIdx(), a3.GetIdx(), c4s[0])
        if tor_hold.getNumTorsions() > 0:
            sys_lig.addForce(tor_hold)

    conf = mol_h.GetConformer(0)
    init_pos = [
        openmm.Vec3(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z) * ounit.angstrom
        for i in range(mol_h.GetNumAtoms())
    ]
    sim = app.Simulation(
        off_mol.to_topology().to_openmm(),
        sys_lig,
        openmm.VerletIntegrator(0.001 * ounit.picoseconds),
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(init_pos)
    sim.minimizeEnergy(maxIterations=500)
    pos_A = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(ounit.angstrom)
    for idx in range(mol_h.GetNumAtoms()):
        conf.SetAtomPosition(idx, pos_A[idx].tolist())

    net_q = float(sum(float(x.m) for x in off_mol.partial_charges))
    return PreparedLigand(
        spec=spec,
        rd_mol=mol_h,
        off_mol=off_mol,
        charge_method=q_method,
        net_charge_e=round(net_q, 4),
    )


def audit_fluorocarbon_torsion(
    prep_lig: PreparedLigand,
    dihedral_indices: Tuple[int, int, int, int],
    step_deg: float = 10.0,
    forcefield_name: str = "openff-2.2.0.offxml",
    use_obc2: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Performs a 1D relaxed OpenFF Sage torsion scan and audits anti/helical stiffness."""
    off_ff = OffForceField(forcefield_name)
    sys_scan = off_ff.create_openmm_system(
        prep_lig.off_mol.to_topology(), charge_from_molecules=[prep_lig.off_mol]
    )
    if use_obc2:
        attach_obc2_solvent(sys_scan, [a.atomic_number for a in prep_lig.off_mol.atoms])

    restraint = openmm.CustomTorsionForce(
        "0.5 * k * min(dtheta, 2*pi - dtheta)^2; dtheta = abs(theta - theta0); pi = 3.141592653589793"
    )
    restraint.addGlobalParameter("k", 1000.0)
    restraint.addGlobalParameter("theta0", 0.0)
    restraint.addTorsion(*dihedral_indices)
    restraint.setForceGroup(1)
    sys_scan.addForce(restraint)

    conf = prep_lig.rd_mol.GetConformer(0)
    init_pos = [
        openmm.Vec3(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z) * ounit.angstrom
        for i in range(prep_lig.rd_mol.GetNumAtoms())
    ]
    sim = app.Simulation(
        prep_lig.off_mol.to_topology().to_openmm(),
        sys_scan,
        openmm.VerletIntegrator(0.001 * ounit.picoseconds),
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(init_pos)

    rows = []
    angles = np.arange(0.0, 360.0 + 1e-6, step_deg)
    for deg in angles:
        rad = math.radians(deg if deg <= 180.0 else deg - 360.0)
        sim.context.setParameter("theta0", rad)
        sim.minimizeEnergy(maxIterations=800)
        e_kcal = sim.context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(
            ounit.kilocalories_per_mole
        )
        rows.append((float(deg), float(e_kcal)))

    min_e = min(r[1] for r in rows)
    df = pd.DataFrame([{"angle_deg": d, "delta_e_kcal_mol": e - min_e} for d, e in rows])
    helical_window = df[(df["angle_deg"] >= 160.0) & (df["angle_deg"] <= 200.0)]
    anti_penalty = float(helical_window["delta_e_kcal_mol"].min())
    summary = {
        "global_min_deg": float(df.loc[df["delta_e_kcal_mol"].idxmin(), "angle_deg"]),
        "barrier_kcal_mol": round(float(df["delta_e_kcal_mol"].max()), 3),
        "helical_160_200_min_kcal_mol": round(anti_penalty, 3),
    }
    return df, summary
