"""Stage-A veto filtering, zero-inflation-free S_total ranking, weight-sensitivity audit, and plots."""

from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pfas_bench.config import RubricSpec

_RDKIT_GEN_SCRIPTS = "/Users/o2satz/.gemini/config/plugins/science/skills/rdkit_general/scripts"
if Path(_RDKIT_GEN_SCRIPTS).exists() and _RDKIT_GEN_SCRIPTS not in sys.path:
    sys.path.append(_RDKIT_GEN_SCRIPTS)

try:
    from compat import PUBLICATION_CYCLE, apply_publication_spines, set_safe_plot_style
except ImportError:
    PUBLICATION_CYCLE = ["#2563EB", "#D97706", "#059669", "#DC2626", "#7C3AED", "#0891B2"]

    def set_safe_plot_style() -> None:
        plt.rcParams.update({"figure.facecolor": "#FFFFFF", "axes.grid": True, "savefig.dpi": 300})

    def apply_publication_spines(ax: Any) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _minmax_norm(values: np.ndarray, floor: float | None = None, ceiling: float | None = None) -> np.ndarray:
    lo = float(np.min(values)) if floor is None else float(floor)
    hi = float(np.max(values)) if ceiling is None else float(ceiling)
    if hi - lo < 1e-9:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _compute_weight_rank1_stability(
    p_hat: np.ndarray,
    e_hat: np.ndarray,
    r_hat: np.ndarray,
    m_hat: np.ndarray,
    pass_mask: np.ndarray,
    rubric: RubricSpec,
    n_draws: int = 500,
) -> np.ndarray:
    """Computes % of +-40% weight perturbations in which each variant ranks #1 on this substrate."""
    n_var = len(p_hat)
    if n_var == 0 or not np.any(pass_mask):
        return np.zeros(n_var, dtype=float)

    rng = np.random.default_rng(20261123)
    base_w = np.array([
        rubric.weight_p_nac,
        rubric.weight_e_proj,
        rubric.weight_r_pose,
        rubric.weight_m_robust,
    ], dtype=float)

    pert = rng.uniform(0.60, 1.40, size=(n_draws, 4)) * base_w[None, :]
    pert = pert / np.sum(pert, axis=1, keepdims=True) * 100.0
    comp_matrix = np.vstack([p_hat, e_hat, r_hat, m_hat])  # (4, n_var)
    scores = pert @ comp_matrix  # (n_draws, n_var)
    scores[:, ~pass_mask] = -1.0

    winners = np.argmax(scores, axis=1)
    win_pct = np.array([100.0 * np.mean(winners == i) if pass_mask[i] else 0.0 for i in range(n_var)], dtype=float)
    return np.round(win_pct, 1)


def rank_variants_table(
    replicate_summaries: Sequence[Dict[str, Any]],
    rubric: RubricSpec,
    fold_gates_by_variant: Dict[str, Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Aggregates replicate trajectories per (variant_id, substrate_id) and computes audit-hardened S_total."""
    df_rep = pd.DataFrame(list(replicate_summaries))
    if df_rep.empty:
        raise ValueError("Cannot rank an empty replicate summary list.")

    grouped_rows: List[Dict[str, Any]] = []
    for (vid, sid), grp in df_rep.groupby(["variant_id", "substrate_id"], sort=False):
        # Support both newly calibrated v0.2 JSONs and legacy v0.1 JSONs (where p_near_nac_pct held ground-state NAC)
        has_ts_col = "p_ts_tight_nac_pct" in grp.columns
        if has_ts_col:
            p_nacs = grp["p_nac_fraction"].to_numpy(dtype=float)
            ts_tight_pct = float(grp["p_ts_tight_nac_pct"].mean())
        else:
            # Legacy v0.1 cached records stored transition-state cutoff in p_nac_pct and ground-state in p_near_nac_pct
            ts_tight_pct = float(grp["p_nac_pct"].mean())
            if "p_near_nac_pct" in grp.columns:
                p_nacs = grp["p_near_nac_pct"].to_numpy(dtype=float) / 100.0
            else:
                p_nacs = grp["p_nac_fraction"].to_numpy(dtype=float)

        mean_pnac = float(np.mean(p_nacs))
        se_pnac = float(np.std(p_nacs, ddof=1) / np.sqrt(len(p_nacs))) if len(p_nacs) > 1 else 0.0
        p_near = float(grp["p_near_nac_pct"].mean()) if "p_near_nac_pct" in grp.columns else 100.0 * mean_pnac

        fg = (fold_gates_by_variant or {}).get(vid, {})
        fold_pass = bool(fg.get("stage_a_fold_pass", True))
        raw_plddt = fg.get("active_site_plddt", None)
        # Treat legacy placeholder 100.0 with zone0_rmsd==0.0 as crystal_3R3V_point_mutant unless source specified
        struct_source = str(fg.get("structure_source", "")).strip()
        z0_rmsd = float(fg.get("zone0_heavy_rmsd_A", 0.0))
        if not struct_source:
            if raw_plddt == 100.0 and z0_rmsd < 0.01:
                struct_source = "crystal_3R3V_point_mutant"
                raw_plddt = None
            elif raw_plddt is not None:
                struct_source = "predicted_af2_or_boltz"
            else:
                struct_source = "crystal_3R3V_point_mutant"

        plddt_factor = (float(raw_plddt) / 100.0) if raw_plddt is not None else 1.0

        # Check nucleophile competence (Stage A Gate 0: D110N lacks catalytic Asp110 carboxylate)
        if "has_anionic_asp110" in grp.columns:
            has_asp110 = bool(grp["has_anionic_asp110"].all())
        else:
            has_asp110 = ("D110N" not in str(vid).upper())

        mean_inv = float(grp["f_inverted_tail"].mean())
        mean_unb = float(grp["f_clamp_unbound"].mean())
        inv_ok = mean_inv <= rubric.max_inverted_tail_in_cradle_fraction
        unb_ok = mean_unb <= rubric.max_clamp_unbound_fraction

        veto_reasons = []
        if not has_asp110:
            veto_reasons.append("no_Asp110_nucleophile")
        if not fold_pass:
            veto_reasons.append("fold_distortion")
        if not inv_ok:
            veto_reasons.append("inverted_tail")
        if not unb_ok:
            veto_reasons.append("clamp_unbound")

        stage_a_pass = len(veto_reasons) == 0
        veto_reason_str = "pass" if stage_a_pass else "+".join(veto_reasons)

        e_rep_vals = grp["nac_E_pull_MV_cm"].to_numpy(dtype=float)
        mean_epull = float(np.mean(e_rep_vals))

        # Clamp legacy PBC-wrapped ca_rmsf_A artifacts (> 3.5 A while clamp stays < 3.2 A) if re-ranking old JSONs
        raw_rmsf = float(grp["ca_rmsf_A"].mean())
        mean_clamp = float(grp["mean_d_clamp_A"].mean()) if "mean_d_clamp_A" in grp.columns else 2.8
        eff_rmsf = 0.55 if (raw_rmsf > 3.5 and mean_clamp < 3.2) else raw_rmsf

        mean_wat = float(grp["f_wat_step2"].mean())
        base_r_pose = 0.5 * (1.0 - min(1.0, eff_rmsf / 2.0)) + 0.5 * mean_wat

        # Eliminate zero-NAC baseline score inflation:
        # Static non-productive poses (mean_pnac == 0 and mean_epull <= 0) get severely damped pose score & 0 robustness
        if mean_pnac > 1e-6:
            r_pose_raw = base_r_pose
            if len(p_nacs) > 1:
                m_robust_raw = plddt_factor * (float(np.min(p_nacs)) / mean_pnac)
            else:
                m_robust_raw = plddt_factor * (1.0 - min(1.0, se_pnac / max(mean_pnac, 0.05)))
        elif np.all(e_rep_vals > 0.0):
            # Pre-organized electrostatic pull achieved across all replicates, awaiting SN2 angle crossing
            r_pose_raw = 0.65 * base_r_pose
            e_consistency = float(np.min(e_rep_vals)) / max(1e-3, float(np.max(e_rep_vals)))
            m_robust_raw = 0.40 * plddt_factor * e_consistency
        else:
            # Neither NAC geometry nor reproducible stabilizing electric field -> zero robustness, minimal pose credit
            r_pose_raw = 0.15 * base_r_pose
            m_robust_raw = 0.0

        # Provenance columns
        n_frames_tot = int(grp["n_frames"].sum()) if "n_frames" in grp.columns else len(grp) * 40
        if "sim_duration_ps" in grp.columns:
            total_prod_ps = round(float(grp["sim_duration_ps"].sum()), 3)
        else:
            # Infer from known 40-frame x 40-step x 1.0 fs head-to-head setup when reading v0.1 JSONs
            total_prod_ps = round(float(n_frames_tot * 0.04), 3)

        grouped_rows.append({
            "variant_id": vid,
            "substrate_id": sid,
            "n_replicates": len(grp),
            "total_prod_ps": total_prod_ps,
            "n_frames_total": n_frames_tot,
            "structure_source": struct_source,
            "active_site_plddt": round(float(raw_plddt), 1) if raw_plddt is not None else np.nan,
            "zone0_rmsd_A": round(z0_rmsd, 3),
            "stage_a_pass": stage_a_pass,
            "veto_reason": veto_reason_str,
            "p_nac_pct": round(100.0 * mean_pnac, 2),
            "p_ts_tight_nac_pct": round(ts_tight_pct, 2),
            "p_near_nac_pct": round(p_near, 2),
            "p_nac_fraction": mean_pnac,
            "nac_E_pull_MV_cm": round(mean_epull, 2),
            "mean_d_attack_A": round(float(grp["mean_d_attack_A"].mean()), 3),
            "mean_ang_SN2_deg": round(float(grp["mean_ang_SN2_deg"].mean()), 1),
            "mean_cradle_hbonds": round(float(grp["mean_cradle_hbonds"].mean()), 2),
            "f_wat_step2": round(mean_wat, 3),
            "f_inverted_tail": round(mean_inv, 3),
            "ca_rmsf_A": round(eff_rmsf, 3),
            "_r_pose_raw": r_pose_raw,
            "_m_robust_raw": m_robust_raw,
        })

    df = pd.DataFrame(grouped_rows)

    s_totals = np.zeros(len(df), dtype=float)
    win_stabilities = np.zeros(len(df), dtype=float)

    for sid, idx_series in df.groupby("substrate_id").groups.items():
        idxs = list(idx_series)
        sub_df = df.loc[idxs]

        # Anchor P_NAC ceiling at 20% ground-state occupancy (or panel max if higher)
        p_hat = np.clip(
            sub_df["p_nac_fraction"].to_numpy(dtype=float) / max(0.20, float(sub_df["p_nac_fraction"].max())),
            0.0,
            1.0,
        )
        e_vals = sub_df["nac_E_pull_MV_cm"].to_numpy(dtype=float)
        # Anchor E_pull floor at 0 MV/cm (destabilizing field <= 0 MV/cm receives 0 electrostatic points!)
        e_hat = _minmax_norm(e_vals, floor=0.0, ceiling=max(35.0, float(np.max(e_vals))))
        r_hat = np.clip(sub_df["_r_pose_raw"].to_numpy(dtype=float), 0.0, 1.0)
        m_hat = np.clip(sub_df["_m_robust_raw"].to_numpy(dtype=float), 0.0, 1.0)

        raw_score = (
            rubric.weight_p_nac * p_hat
            + rubric.weight_e_proj * e_hat
            + rubric.weight_r_pose * r_hat
            + rubric.weight_m_robust * m_hat
        )
        pass_mask = sub_df["stage_a_pass"].to_numpy(dtype=bool)
        s_totals[idxs] = np.where(pass_mask, np.round(raw_score, 2), 0.0)
        win_stabilities[idxs] = _compute_weight_rank1_stability(
            p_hat, e_hat, r_hat, m_hat, pass_mask, rubric
        )

    df["S_total"] = s_totals
    df["top1_weight_stability_pct"] = win_stabilities
    df = df.drop(columns=["_r_pose_raw", "_m_robust_raw", "p_nac_fraction"])
    df = df.sort_values(by=["substrate_id", "stage_a_pass", "S_total"], ascending=[True, False, False]).reset_index(drop=True)
    return df


def render_ranking_plot(df_ranked: pd.DataFrame, out_png: str | Path) -> Path:
    """Renders a substrate-faceted publication leaderboard of S_total and Scissile E_pull."""
    out_p = Path(out_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    set_safe_plot_style()

    substrates = list(dict.fromkeys(df_ranked["substrate_id"].tolist()))
    if "PFBA" in substrates:
        substrates = ["PFBA"] + [s for s in substrates if s != "PFBA"]

    n_rows = len(substrates)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(11.8, max(3.6, 2.9 * n_rows)),
        layout="constrained",
        dpi=300,
        squeeze=False,
    )

    for r_idx, sid in enumerate(substrates):
        sub = df_ranked[df_ranked["substrate_id"] == sid].reset_index(drop=True)
        y = np.arange(len(sub))
        labels = [row.variant_id for row in sub.itertuples()]

        ax_s = axes[r_idx, 0]
        colors_s = [PUBLICATION_CYCLE[2] if row.stage_a_pass else "#94A3B8" for row in sub.itertuples()]
        bars_s = ax_s.barh(y, sub["S_total"], color=colors_s, height=0.62)
        ax_s.set_yticks(y, labels)
        ax_s.invert_yaxis()
        ax_s.set_xlim(0, 108)
        ax_s.set_xlabel("Composite Rubric Score S_total (0–100)")
        ax_s.set_title(f"{sid}: Variant Pre-Organization Score (v0.2 Calibrated)")
        for b_idx, bar in enumerate(bars_s):
            row = sub.iloc[b_idx]
            val = float(row["S_total"])
            if bool(row["stage_a_pass"]):
                tag = f" {val:.1f} (NAC={row['p_nac_pct']:.1f}%)"
            else:
                tag = f" 0.0 [VETO: {row['veto_reason']}]"
            ax_s.text(max(val, 0.8) + 1.2, bar.get_y() + bar.get_height() / 2.0, tag, va="center", fontsize=8.0, fontweight="bold")
        apply_publication_spines(ax_s)

        ax_e = axes[r_idx, 1]
        e_vals = sub["nac_E_pull_MV_cm"].to_numpy(dtype=float)
        colors_e = [PUBLICATION_CYCLE[0] if v >= 0 else "#DC2626" for v in e_vals]
        ax_e.barh(y, e_vals, color=colors_e, height=0.62)
        ax_e.axvline(0.0, color="#475569", linestyle="--", linewidth=1.1)
        ax_e.set_yticks(y, labels)
        ax_e.invert_yaxis()
        e_pad = max(15.0, 0.22 * (float(np.max(np.abs(e_vals))) + 10.0))
        ax_e.set_xlim(min(-25.0, float(np.min(e_vals)) - e_pad), max(45.0, float(np.max(e_vals)) + e_pad))
        ax_e.set_xlabel("Scissile Electric Field E_pull (MV/cm)")
        ax_e.set_title(f"{sid}: Active-Site Cα–F Polarizing Field")
        for b_idx, v in enumerate(e_vals):
            offset = 1.8 if v >= 0 else -1.8
            ha = "left" if v >= 0 else "right"
            ax_e.text(v + offset, y[b_idx], f"{v:+.1f}", va="center", ha=ha, fontsize=8.5)
        apply_publication_spines(ax_e)

    fig.savefig(out_p)
    plt.close(fig)
    return out_p
