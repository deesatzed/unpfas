"""Stage-A veto filtering, composite S_total ranking, and publication-styled plots."""

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


def rank_variants_table(
    replicate_summaries: Sequence[Dict[str, Any]],
    rubric: RubricSpec,
    fold_gates_by_variant: Dict[str, Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Aggregates replicate trajectories per (variant_id, substrate_id) and computes S_total in [0, 100]."""
    df_rep = pd.DataFrame(list(replicate_summaries))
    if df_rep.empty:
        raise ValueError("Cannot rank an empty replicate summary list.")

    grouped_rows: List[Dict[str, Any]] = []
    for (vid, sid), grp in df_rep.groupby(["variant_id", "substrate_id"], sort=False):
        p_nacs = grp["p_nac_fraction"].to_numpy(dtype=float)
        mean_pnac = float(np.mean(p_nacs))
        se_pnac = float(np.std(p_nacs, ddof=1) / np.sqrt(len(p_nacs))) if len(p_nacs) > 1 else 0.0

        fg = (fold_gates_by_variant or {}).get(vid, {"active_site_plddt": 95.0, "zone0_heavy_rmsd_A": 0.25, "stage_a_fold_pass": True})
        fold_pass = bool(fg.get("stage_a_fold_pass", True))
        plddt = float(fg.get("active_site_plddt", 95.0))
        z0_rmsd = float(fg.get("zone0_heavy_rmsd_A", 0.25))

        mean_inv = float(grp["f_inverted_tail"].mean())
        mean_unb = float(grp["f_clamp_unbound"].mean())
        pose_pass = (mean_inv <= rubric.max_inverted_tail_in_cradle_fraction) and (
            mean_unb <= rubric.max_clamp_unbound_fraction
        )
        stage_a_pass = fold_pass and pose_pass

        mean_rmsf = float(grp["ca_rmsf_A"].mean())
        mean_wat = float(grp["f_wat_step2"].mean())
        r_pose_raw = 0.5 * (1.0 - min(1.0, mean_rmsf / 2.0)) + 0.5 * mean_wat

        # Robustness across replicates or environmental matrix conditions
        if len(p_nacs) > 1 and mean_pnac > 1e-6:
            min_ratio = float(np.min(p_nacs)) / mean_pnac
            m_robust_raw = (plddt / 100.0) * min_ratio
        else:
            m_robust_raw = (plddt / 100.0) * (1.0 - min(1.0, se_pnac / max(mean_pnac, 0.05)))

        p_near = float(grp["p_near_nac_pct"].mean()) if "p_near_nac_pct" in grp.columns else 100.0 * mean_pnac

        grouped_rows.append({
            "variant_id": vid,
            "substrate_id": sid,
            "n_replicates": len(grp),
            "stage_a_pass": stage_a_pass,
            "active_site_plddt": round(plddt, 1),
            "zone0_rmsd_A": round(z0_rmsd, 3),
            "p_nac_pct": round(100.0 * mean_pnac, 2),
            "p_near_nac_pct": round(p_near, 2),
            "p_nac_fraction": mean_pnac,
            "nac_E_pull_MV_cm": round(float(grp["nac_E_pull_MV_cm"].mean()), 2),
            "mean_d_attack_A": round(float(grp["mean_d_attack_A"].mean()), 3),
            "mean_ang_SN2_deg": round(float(grp["mean_ang_SN2_deg"].mean()), 1),
            "mean_cradle_hbonds": round(float(grp["mean_cradle_hbonds"].mean()), 2),
            "f_wat_step2": round(mean_wat, 3),
            "f_inverted_tail": round(mean_inv, 3),
            "ca_rmsf_A": round(mean_rmsf, 3),
            "_r_pose_raw": r_pose_raw,
            "_m_robust_raw": m_robust_raw,
        })

    df = pd.DataFrame(grouped_rows)

    # Normalize per substrate (or across full panel when single-point testing)
    s_totals = np.zeros(len(df), dtype=float)
    for sid, idx_series in df.groupby("substrate_id").groups.items():
        idxs = list(idx_series)
        sub_df = df.loc[idxs]

        p_hat = np.clip(sub_df["p_nac_fraction"].to_numpy(dtype=float) / max(0.25, float(sub_df["p_nac_fraction"].max())), 0.0, 1.0)
        e_vals = sub_df["nac_E_pull_MV_cm"].to_numpy(dtype=float)
        # Anchor electric field normalization: <= -20 MV/cm -> 0, >= +80 MV/cm (or panel max) -> 1
        e_hat = _minmax_norm(e_vals, floor=-20.0, ceiling=max(60.0, float(np.max(e_vals))))
        r_hat = np.clip(sub_df["_r_pose_raw"].to_numpy(dtype=float), 0.0, 1.0)
        m_hat = np.clip(sub_df["_m_robust_raw"].to_numpy(dtype=float), 0.0, 1.0)

        raw_score = (
            rubric.weight_p_nac * p_hat
            + rubric.weight_e_proj * e_hat
            + rubric.weight_r_pose * r_hat
            + rubric.weight_m_robust * m_hat
        )
        # Stage A hard veto zeroes composite score while keeping diagnostic columns transparent
        pass_mask = sub_df["stage_a_pass"].to_numpy(dtype=bool)
        s_totals[idxs] = np.where(pass_mask, np.round(raw_score, 2), 0.0)

    df["S_total"] = s_totals
    df = df.drop(columns=["_r_pose_raw", "_m_robust_raw", "p_nac_fraction"])
    df = df.sort_values(by=["substrate_id", "stage_a_pass", "S_total"], ascending=[True, False, False]).reset_index(drop=True)
    return df


def render_ranking_plot(df_ranked: pd.DataFrame, out_png: str | Path) -> Path:
    """Renders a substrate-faceted publication leaderboard of S_total and Scissile E_pull."""
    out_p = Path(out_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    set_safe_plot_style()

    substrates = list(dict.fromkeys(df_ranked["substrate_id"].tolist()))
    # Show PFBA at top if present
    if "PFBA" in substrates:
        substrates = ["PFBA"] + [s for s in substrates if s != "PFBA"]

    n_rows = len(substrates)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(11.5, max(3.6, 2.9 * n_rows)),
        layout="constrained",
        dpi=300,
        squeeze=False,
    )

    for r_idx, sid in enumerate(substrates):
        sub = df_ranked[df_ranked["substrate_id"] == sid].reset_index(drop=True)
        y = np.arange(len(sub))
        labels = [row.variant_id for row in sub.itertuples()]

        # Left column: S_total (0-100) with Stage-A Veto / Dead-Control callouts
        ax_s = axes[r_idx, 0]
        colors_s = [PUBLICATION_CYCLE[2] if row.stage_a_pass else "#94A3B8" for row in sub.itertuples()]
        bars_s = ax_s.barh(y, sub["S_total"], color=colors_s, height=0.62)
        ax_s.set_yticks(y, labels)
        ax_s.invert_yaxis()
        ax_s.set_xlim(0, 108)
        ax_s.set_xlabel("Composite Rubric Score S_total (0–100)")
        ax_s.set_title(f"{sid}: Variant Pre-Organization Score (Explicit TIP3P-FB + PME)")
        for b_idx, bar in enumerate(bars_s):
            row = sub.iloc[b_idx]
            val = float(row["S_total"])
            tag = f" {val:.1f}" if bool(row["stage_a_pass"]) else f" VETO (inv={100*row['f_inverted_tail']:.0f}%)"
            ax_s.text(max(val, 1.0) + 1.2, bar.get_y() + bar.get_height() / 2.0, tag, va="center", fontsize=8.5, fontweight="bold")
        apply_publication_spines(ax_s)

        # Right column: Scissile E_pull (MV/cm)
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
