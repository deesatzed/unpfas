"""Typed YAML configuration loader and Zone-0 mutation validator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass(frozen=True)
class SubstrateSpec:
    id: str
    name: str
    role: str
    smiles: str
    formal_charge: int = -1


@dataclass(frozen=True)
class VariantSpec:
    id: str
    mutations: Dict[int, str]
    role: str


@dataclass(frozen=True)
class RubricSpec:
    min_active_site_plddt: float = 85.0
    max_zone0_heavy_rmsd_A: float = 0.80
    max_inverted_tail_in_cradle_fraction: float = 0.50
    max_clamp_unbound_fraction: float = 0.30
    max_attack_dist_A: float = 3.30
    min_attack_angle_deg: float = 155.0
    max_attack_angle_deg: float = 180.0
    min_cradle_hbonds: int = 2
    max_cradle_H_F_dist_A: float = 2.50
    max_clamp_N_O_dist_A: float = 3.40
    max_His280_NE2_water_O_dist_A: float = 6.00
    max_Asp110_CG_water_O_dist_A: float = 5.80
    weight_p_nac: float = 35.0
    weight_e_proj: float = 25.0
    weight_r_pose: float = 20.0
    weight_m_robust: float = 20.0


@dataclass
class BenchmarkConfig:
    raw: Dict[str, Any]
    root_dir: Path
    cif_holo_path: Path
    cif_apo_path: Path
    zone0_residues: List[int]
    charge_method_priority: List[str]
    torsion_sweep_step_deg: float
    substrates: Dict[str, SubstrateSpec] = field(default_factory=dict)
    variants: Dict[str, VariantSpec] = field(default_factory=dict)
    rubric: RubricSpec = field(default_factory=RubricSpec)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, root_dir: str | Path | None = None) -> "BenchmarkConfig":
        ypath = Path(yaml_path).resolve()
        rdir = Path(root_dir).resolve() if root_dir else ypath.parent.parent
        with open(ypath, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        scaf = data["scaffold"]
        cif_holo = rdir / scaf.get("cif_holo_path", "data/structures/pdb_00003r3v.cif")
        cif_apo = rdir / scaf.get("cif_apo_path", "data/structures/pdb_00003r3u.cif")

        z0_cfg = data["zones"]["zone_0_immutable_core"]
        z0_all = sorted(
            set(z0_cfg["catalytic_triad"])
            | set(z0_cfg["halide_cradle"])
            | set(z0_cfg["carboxylate_clamp"])
        )

        lig_cfg = data["ligands"]
        subs = {}
        for item in lig_cfg["substrates"]:
            spec = SubstrateSpec(
                id=item["id"],
                name=item["name"],
                role=item["role"],
                smiles=item["smiles"],
                formal_charge=int(item.get("formal_charge", -1)),
            )
            subs[spec.id] = spec

        vars_map = {}
        for vitem in data["variants"]:
            mut_dict = {int(k): str(v).upper() for k, v in vitem.get("mutations", {}).items()}
            role = str(vitem.get("role", "engineered"))
            # Enforce Zone-0 immutability unless explicitly marked as a negative control
            z0_hits = set(mut_dict.keys()) & set(z0_all)
            if z0_hits and "negative_control" not in role:
                raise ValueError(
                    f"Variant {vitem['id']} mutates immutable Zone-0 residue(s) {sorted(z0_hits)} "
                    f"without role='...negative_control...'."
                )
            vspec = VariantSpec(id=vitem["id"], mutations=mut_dict, role=role)
            vars_map[vspec.id] = vspec

        rcfg = data["scoring_rubric"]
        sa = rcfg["stage_a_vetoes"]
        nc = rcfg["nac_criteria"]
        sw = rcfg["step2_water_check"]
        wt = rcfg["weights"]
        rubric = RubricSpec(
            min_active_site_plddt=float(sa["min_active_site_plddt"]),
            max_zone0_heavy_rmsd_A=float(sa["max_zone0_heavy_rmsd_A"]),
            max_inverted_tail_in_cradle_fraction=float(sa["max_inverted_tail_in_cradle_fraction"]),
            max_clamp_unbound_fraction=float(sa["max_clamp_unbound_fraction"]),
            max_attack_dist_A=float(nc["max_attack_dist_Asp110_Od_Ca_A"]),
            min_attack_angle_deg=float(nc["min_attack_angle_deg"]),
            max_attack_angle_deg=float(nc["max_attack_angle_deg"]),
            min_cradle_hbonds=int(nc["min_cradle_hbonds"]),
            max_cradle_H_F_dist_A=float(nc["max_cradle_H_F_dist_A"]),
            max_clamp_N_O_dist_A=float(nc["max_clamp_N_O_dist_A"]),
            max_His280_NE2_water_O_dist_A=float(sw["max_His280_NE2_water_O_dist_A"]),
            max_Asp110_CG_water_O_dist_A=float(sw["max_Asp110_CG_water_O_dist_A"]),
            weight_p_nac=float(wt["p_nac"]),
            weight_e_proj=float(wt["e_proj_scissile_cf"]),
            weight_r_pose=float(wt["r_pose_and_step2_water"]),
            weight_m_robust=float(wt["m_matrix_robustness"]),
        )

        return cls(
            raw=data,
            root_dir=rdir,
            cif_holo_path=cif_holo,
            cif_apo_path=cif_apo,
            zone0_residues=z0_all,
            charge_method_priority=list(lig_cfg["charge_method_priority"]),
            torsion_sweep_step_deg=float(lig_cfg.get("torsion_sweep_step_deg", 15.0)),
            substrates=subs,
            variants=vars_map,
            rubric=rubric,
        )
