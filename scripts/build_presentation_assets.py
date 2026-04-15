#!/usr/bin/env python3
"""Build a rich, polished presentation asset pack from current IF2RNA results."""

from pathlib import Path
import pickle
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "presentation_assets" / "2026-04-06"
OUT.mkdir(parents=True, exist_ok=True)


MODEL_COLORS = {
    "attention_mil": "#0B6E4F",
    "vis_main": "#126782",
    "vit_baseline": "#F07167",
    "meanpool_ridge_baseline": "#9C6644",
    "vis": "#126782",
    "vit": "#F07167",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "#F8F7F4",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D8D6D0",
        "axes.grid": False,
        "grid.color": "#E7E5DF",
        "grid.alpha": 1.0,
        "grid.linestyle": "-",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 240,
    })


def annotate_vertical_bars(ax, decimals: int = 3, y_offset_frac: float = 0.01):
    heights = []
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if np.isfinite(h):
                heights.append(float(h))
    if not heights:
        return

    y0, y1 = ax.get_ylim()
    hmin = min(heights)
    hmax = max(heights)
    span = max(y1 - y0, 1e-8)
    pad = max(span * 0.08, (hmax - hmin) * 0.10, 0.01)

    if hmin >= 0:
        ax.set_ylim(min(0.0, y0), max(y1, hmax + pad))
    else:
        ax.set_ylim(min(y0, hmin - pad), max(y1, hmax + pad))

    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * y_offset_frac
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if np.isnan(h):
                continue
            y_text = h + offset if h >= 0 else h - offset
            va = "bottom" if h >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                f"{h:.{decimals}f}",
                ha="center",
                va=va,
                fontsize=10,
                color="#222222",
                clip_on=False,
            )


def annotate_horizontal_bars(ax, decimals: int = 3, x_offset_frac: float = 0.012):
    widths = []
    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            if np.isfinite(w):
                widths.append(float(w))
    if not widths:
        return

    x0, x1 = ax.get_xlim()
    wmax = max(widths)
    span = max(x1 - x0, 1e-8)
    pad = max(span * 0.10, wmax * 0.05, 0.8)
    ax.set_xlim(min(0.0, x0), max(x1, wmax + pad))

    x0, x1 = ax.get_xlim()
    offset = (x1 - x0) * x_offset_frac
    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            if np.isnan(w):
                continue
            ax.text(
                w + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.{decimals}f}",
                ha="left",
                va="center",
                fontsize=9,
                color="#222222",
                clip_on=False,
            )


def save_df(df: pd.DataFrame, name: str):
    out = OUT / name
    df.to_csv(out, index=False)
    return out


def set_legend_above(ax, ncol: int = 1, y: float = 1.09):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=ncol,
            frameon=True,
            borderaxespad=0.3,
        )


def render_table_png(df: pd.DataFrame, title: str, out_name: str, max_rows: int = 18):
    view = df.copy().head(max_rows)
    ncols = len(view.columns)
    fig_w = min(24, max(12, 2.25 * ncols))
    fig_h = max(2.8, 0.50 * (len(view) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=10, fontweight="bold")

    # Format numeric columns for readability.
    formatted = view.copy()
    for c in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[c]):
            formatted[c] = formatted[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    # Keep table cells compact and prevent overflow from very long labels.
    for c in formatted.columns:
        if pd.api.types.is_object_dtype(formatted[c]):
            formatted[c] = formatted[c].astype(str).map(lambda s: textwrap.shorten(s.replace("_", " "), width=28, placeholder="..."))

    col_labels = [textwrap.fill(str(c), width=16, break_long_words=False) for c in formatted.columns]

    table = ax.table(
        cellText=formatted.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.28)
    try:
        table.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        pass

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D6D0")
        if r == 0:
            cell.set_facecolor("#EDEAE3")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#FBFAF8")

    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)


def load_test_results(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    split_keys = sorted([k for k in obj.keys() if str(k).startswith("split_")])

    preds_all, real_all = [], []
    for k in split_keys:
        d = obj[k]
        pred = np.asarray(d.get("preds", []))
        real = np.asarray(d.get("real", []))
        if pred.size == 0 or real.size == 0:
            continue
        preds_all.append(pred)
        real_all.append(real)

    if not preds_all:
        raise RuntimeError(f"No usable split data in {pkl_path}")

    preds = np.vstack(preds_all)
    real = np.vstack(real_all)

    corrs = []
    for i in range(preds.shape[1]):
        m = ~(np.isnan(preds[:, i]) | np.isnan(real[:, i]))
        if m.sum() > 1 and np.std(preds[m, i]) > 0 and np.std(real[m, i]) > 0:
            corrs.append(pearsonr(preds[m, i], real[m, i])[0])
    corrs = np.asarray(corrs, dtype=float)

    summary = {
        "n_samples": int(preds.shape[0]),
        "n_genes": int(preds.shape[1]),
        "n_valid_genes": int(corrs.size),
        "mean_gene_correlation": float(np.nanmean(corrs)),
        "median_gene_correlation": float(np.nanmedian(corrs)),
        "mae": float(np.mean(np.abs(preds - real))),
        "rmse": float(np.sqrt(np.mean((preds - real) ** 2))),
        "genes_r_gt_0_3": int(np.sum(corrs > 0.3)),
        "genes_r_gt_0_5": int(np.sum(corrs > 0.5)),
    }
    return summary, corrs


def series_color(name: str):
    return MODEL_COLORS.get(name, "#4C4C4C")


def pretty_label(name: str):
    mapping = {
        "attention_mil": "Attention MIL",
        "vis_main": "ViS",
        "vit_baseline": "ViT",
        "meanpool_ridge_baseline": "MeanPool Ridge",
        "vis": "ViS",
        "vit": "ViT",
    }
    return mapping.get(name, str(name).replace("_", " "))


def main():
    set_style()

    # ----------------------------
    # Build core tables.
    # ----------------------------
    phase2 = pd.read_csv(ROOT / "results" / "phase3_summary" / "phase2_model_comparison_snapshot.csv")
    amil_stats, amil_corr = load_test_results(ROOT / "results" / "if2rna_models" / "phase2_attention_mil" / "test_results.pkl")
    vis_stats, vis_corr = load_test_results(ROOT / "results" / "if2rna_models" / "phase2_vis_main" / "test_results.pkl")
    vit_stats, vit_corr = load_test_results(ROOT / "results" / "if2rna_models" / "phase2_vit_baseline" / "test_results.pkl")
    mean_stats, mean_corr = load_test_results(ROOT / "results" / "if2rna_models" / "phase2_meanpool_baseline" / "test_results.pkl")

    amil_row = {
        "model": "attention_mil",
        "n_samples": amil_stats["n_samples"],
        "n_genes": amil_stats["n_genes"],
        "mean_gene_correlation": amil_stats["mean_gene_correlation"],
        "median_gene_correlation": amil_stats["median_gene_correlation"],
        "mae": amil_stats["mae"],
        "rmse": amil_stats["rmse"],
        "genes_r_gt_0_3": amil_stats["genes_r_gt_0_3"],
        "genes_r_gt_0_5": amil_stats["genes_r_gt_0_5"],
    }

    phase2_ext = pd.concat(
        [phase2.drop(columns=[c for c in ["delta_mean_corr_vs_best"] if c in phase2.columns]), pd.DataFrame([amil_row])],
        ignore_index=True,
    )
    phase2_ext = phase2_ext.sort_values("mean_gene_correlation", ascending=False).reset_index(drop=True)
    best = phase2_ext["mean_gene_correlation"].max()
    phase2_ext["delta_mean_corr_vs_best"] = phase2_ext["mean_gene_correlation"] - best
    save_df(phase2_ext, "table_id_main_with_amil.csv")
    render_table_png(phase2_ext, "Main ID Benchmark (with Attention-MIL)", "table_id_main_with_amil.png")

    organ_agg = pd.read_csv(ROOT / "results" / "phase3_summary" / "ood_aggregate_model_comparison.csv")
    organ_agg["ood_type"] = "heldout_organ"

    slide_rows = []
    slide_sources = {
        "meanpool_ridge_baseline": ROOT / "results" / "phase3_ood" / "heldout_slide_meanpool" / "heldout_slide_aggregate.csv",
        "vit": ROOT / "results" / "phase3_ood" / "heldout_slide_vit" / "heldout_slide_aggregate.csv",
        "vis": ROOT / "results" / "phase3_ood" / "heldout_slide_vis" / "heldout_slide_aggregate.csv",
    }
    for model, p in slide_sources.items():
        if p.exists():
            d = pd.read_csv(p)
            slide_rows.append({
                "model": model,
                "ood_mean_of_means": float(d["ood_mean_of_means"].iloc[0]),
                "ood_mean_of_medians": float(d["ood_mean_of_medians"].iloc[0]),
                "ood_mean_mae": float(d["ood_mean_mae"].iloc[0]),
                "ood_mean_rmse": float(d["ood_mean_rmse"].iloc[0]),
                "ood_type": "heldout_slide",
            })
        else:
            slide_rows.append({
                "model": model,
                "ood_mean_of_means": np.nan,
                "ood_mean_of_medians": np.nan,
                "ood_mean_mae": np.nan,
                "ood_mean_rmse": np.nan,
                "ood_type": "heldout_slide",
            })

    ood_combo = pd.concat(
        [organ_agg[["model", "ood_mean_of_means", "ood_mean_of_medians", "ood_mean_mae", "ood_mean_rmse", "ood_type"]], pd.DataFrame(slide_rows)],
        ignore_index=True,
    )
    ood_combo_display = ood_combo.copy()
    metric_cols = ["ood_mean_of_means", "ood_mean_of_medians", "ood_mean_mae", "ood_mean_rmse"]
    for c in metric_cols:
        ood_combo_display[c] = ood_combo_display[c].where(ood_combo_display[c].notna(), "in progress")
    save_df(ood_combo_display, "table_ood_aggregate.csv")
    render_table_png(ood_combo_display, "OOD Aggregate Summary", "table_ood_aggregate.png")

    per_organ = pd.read_csv(ROOT / "results" / "phase3_summary" / "phase3_per_organ_model_comparison.csv")
    save_df(per_organ, "table_per_organ_specialization.csv")
    render_table_png(per_organ, "Per-Organ Specialization", "table_per_organ_specialization.png")

    run_status = pd.DataFrame([
        {"run": "attention_mil", "status": "completed", "details": "Strong ID result"},
        {"run": "bag_cap_100", "status": "completed", "details": "Ablation baseline-equivalent cap"},
        {"run": "bag_cap_64", "status": "failed", "details": "ViS patch-count mismatch"},
        {"run": "heldout_slide_vit", "status": "completed", "details": "OOD slide summary available"},
        {"run": "heldout_slide_vis", "status": "timeout", "details": "20h time limit reached"},
        {"run": "elastic_net", "status": "oom", "details": "Memory exceeded"},
    ])
    save_df(run_status, "table_run_status_and_blockers.csv")
    render_table_png(run_status, "Run Status and Blockers", "table_run_status_and_blockers.png")

    # Slide-level summaries for additional plots/tables.
    slide_mean = pd.read_csv(ROOT / "results" / "phase3_ood" / "heldout_slide_meanpool" / "heldout_slide_summary.csv")
    slide_vit = pd.read_csv(ROOT / "results" / "phase3_ood" / "heldout_slide_vit" / "heldout_slide_summary.csv")
    slide_comp = slide_mean[["heldout_slide", "mean_gene_correlation", "mae", "rmse"]].rename(
        columns={
            "mean_gene_correlation": "meanpool_mean_corr",
            "mae": "meanpool_mae",
            "rmse": "meanpool_rmse",
        }
    ).merge(
        slide_vit[["heldout_slide", "mean_gene_correlation", "mae", "rmse"]].rename(
            columns={
                "mean_gene_correlation": "vit_mean_corr",
                "mae": "vit_mae",
                "rmse": "vit_rmse",
            }
        ),
        on="heldout_slide",
        how="outer",
    )
    slide_comp["delta_vit_minus_meanpool_corr"] = slide_comp["vit_mean_corr"] - slide_comp["meanpool_mean_corr"]
    save_df(slide_comp, "table_slide_level_model_comparison.csv")
    render_table_png(slide_comp.sort_values("delta_vit_minus_meanpool_corr", ascending=False), "Held-out Slide: ViT vs Meanpool", "table_slide_level_model_comparison.png")

    # ----------------------------
    # Figures (many).
    # ----------------------------
    # 1) ID mean corr.
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [pretty_label(m) for m in phase2_ext["model"]]
    ax.bar(labels, phase2_ext["mean_gene_correlation"], width=0.56, color=[series_color(m) for m in phase2_ext["model"]])
    ax.set_title("ID Benchmark: Mean Gene-wise Correlation")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, phase2_ext["mean_gene_correlation"].max() * 1.16)
    annotate_vertical_bars(ax, decimals=3)
    ax.grid(False)
    plt.xticks(rotation=12)
    plt.tight_layout()
    fig.savefig(OUT / "fig_01_id_mean_corr_bar.png")
    # Compatibility names requested in prior assets
    fig.savefig(OUT / "fig_id_model_comparison.png")
    fig.savefig(OUT / "fig_id_model_comparision.png")
    plt.close(fig)

    # 2) ID median corr.
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [pretty_label(m) for m in phase2_ext["model"]]
    ax.bar(labels, phase2_ext["median_gene_correlation"], width=0.56, color=[series_color(m) for m in phase2_ext["model"]])
    ax.set_title("ID Benchmark: Median Gene-wise Correlation")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, phase2_ext["median_gene_correlation"].max() * 1.15)
    annotate_vertical_bars(ax, decimals=3)
    ax.grid(False)
    plt.xticks(rotation=12)
    plt.tight_layout()
    fig.savefig(OUT / "fig_02_id_median_corr_bar.png")
    plt.close(fig)

    # 3) ID error metrics.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(phase2_ext))
    w = 0.32
    ax.bar(x - w / 2, phase2_ext["mae"], width=w, label="MAE", color="#355070")
    ax.bar(x + w / 2, phase2_ext["rmse"], width=w, label="RMSE", color="#B56576")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(m) for m in phase2_ext["model"]], rotation=15)
    ax.set_title("ID Benchmark: Error Metrics")
    ax.grid(False)
    annotate_vertical_bars(ax, decimals=3)
    set_legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_03_id_error_bars.png")
    plt.close(fig)

    # 4) Threshold counts.
    fig, ax = plt.subplots(figsize=(10, 5))
    ylabels = [pretty_label(m) for m in phase2_ext["model"]]
    ax.barh(ylabels, phase2_ext["genes_r_gt_0_3"], color="#6A994E", label="genes r > 0.3")
    ax.barh(ylabels, phase2_ext["genes_r_gt_0_5"], color="#386641", label="genes r > 0.5")
    ax.set_title("ID: Number of Well-Predicted Genes")
    ax.grid(False)
    annotate_horizontal_bars(ax, decimals=0)
    set_legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_04_id_threshold_counts.png")
    plt.close(fig)

    # 5) Corr vs MAE scatter.
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in phase2_ext.iterrows():
        ax.scatter(row["mean_gene_correlation"], row["mae"], s=130, color=series_color(row["model"]))
        ax.text(row["mean_gene_correlation"] + 0.003, row["mae"] + 0.002, pretty_label(row["model"]), fontsize=10)
    ax.set_title("ID Trade-off: Correlation vs MAE")
    ax.set_xlabel("Mean gene-wise r")
    ax.set_ylabel("MAE")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(OUT / "fig_05_id_scatter_corr_vs_mae.png")
    plt.close(fig)

    # 6) Per-gene correlation distributions (boxplot).
    corr_df = pd.DataFrame({
        "attention_mil": amil_corr,
        "vis_main": vis_corr,
        "vit_baseline": vit_corr,
        "meanpool_ridge_baseline": mean_corr,
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot([corr_df[c].dropna().values for c in corr_df.columns], tick_labels=list(corr_df.columns), patch_artist=True)
    for patch, c in zip(bp["boxes"], [series_color(c) for c in corr_df.columns]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_title("Per-Gene Correlation Distribution by Model")
    ax.set_ylabel("Pearson r")
    ax.grid(False)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(OUT / "fig_06_gene_corr_boxplot.png")
    plt.close(fig)

    # 7) CDF curves.
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in corr_df.columns:
        arr = np.sort(corr_df[c].dropna().values)
        y = np.linspace(0, 1, len(arr))
        ax.plot(arr, y, label=c, color=series_color(c), linewidth=2)
    ax.axvline(0.3, color="#555", linestyle="--", linewidth=1)
    ax.set_title("CDF of Gene-wise Correlations")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Cumulative fraction of genes")
    ax.grid(False)
    set_legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_07_gene_corr_cdf.png")
    plt.close(fig)

    # 8) Ranked gene correlation curves.
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in corr_df.columns:
        arr = np.sort(corr_df[c].dropna().values)[::-1]
        ax.plot(np.arange(1, len(arr) + 1), arr, label=c, color=series_color(c), linewidth=1.6)
    ax.set_title("Ranked Gene-wise Correlation Curves")
    ax.set_xlabel("Gene rank")
    ax.set_ylabel("Pearson r")
    ax.grid(False)
    set_legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_08_gene_corr_rank_curves.png")
    plt.close(fig)

    # 9) OOD aggregate mean corr by split.
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = ood_combo.pivot_table(index="model", columns="ood_type", values="ood_mean_of_means", aggfunc="first")
    pivot = pivot.reindex([m for m in ["vis", "vit", "meanpool_ridge_baseline"] if m in pivot.index])
    pivot.plot(kind="bar", ax=ax, color=["#7B8CDE", "#56CFE1"], width=0.58)
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_title("OOD Mean Correlation: Held-out Organ vs Held-out Slide")
    ax.set_ylabel("OOD mean of mean r")
    ax.grid(False)
    annotate_vertical_bars(ax, decimals=3)
    set_legend_above(ax, ncol=2)
    plt.xticks(rotation=0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_09_ood_corr_by_split.png")
    plt.close(fig)

    # 10) OOD MAE/RMSE by split.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    pivot_mae = ood_combo.pivot_table(index="model", columns="ood_type", values="ood_mean_mae", aggfunc="first")
    pivot_rmse = ood_combo.pivot_table(index="model", columns="ood_type", values="ood_mean_rmse", aggfunc="first")
    pivot_mae.reindex(["vis", "vit", "meanpool_ridge_baseline"]).plot(kind="bar", ax=axes[0], color=["#FFAFCC", "#CDB4DB"], width=0.58)
    axes[0].set_title("OOD Mean MAE")
    axes[0].set_ylabel("MAE")
    axes[0].grid(False)
    annotate_vertical_bars(axes[0], decimals=3)
    set_legend_above(axes[0], ncol=2)
    axes[0].tick_params(axis="x", rotation=0)
    pivot_rmse.reindex(["vis", "vit", "meanpool_ridge_baseline"]).plot(kind="bar", ax=axes[1], color=["#FFD6A5", "#CAFFBF"], width=0.58)
    axes[1].set_title("OOD Mean RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(False)
    annotate_vertical_bars(axes[1], decimals=3)
    set_legend_above(axes[1], ncol=2)
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_10_ood_error_by_split.png")
    plt.close(fig)

    # 11) Per-organ heatmap.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    heat = per_organ.pivot_table(index="organ", columns="model", values="mean_correlation", aggfunc="mean")
    heat = heat.reindex(columns=[c for c in ["vis", "vit"] if c in heat.columns])
    im = ax.imshow(heat.values, cmap="coolwarm", vmin=-0.15, vmax=0.15)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Per-Organ Specialization Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(OUT / "fig_11_per_organ_heatmap.png")
    plt.close(fig)

    # 12) Per-organ grouped bars.
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    orgs = sorted(per_organ["organ"].unique())
    vis_vals = []
    vit_vals = []
    for o in orgs:
        vis_vals.append(float(per_organ[(per_organ["organ"] == o) & (per_organ["model"] == "vis")]["mean_correlation"].iloc[0]))
        vit_vals.append(float(per_organ[(per_organ["organ"] == o) & (per_organ["model"] == "vit")]["mean_correlation"].iloc[0]))
    x = np.arange(len(orgs))
    w = 0.31
    ax.bar(x - w / 2, vis_vals, width=w, label="ViS", color=series_color("vis"))
    ax.bar(x + w / 2, vit_vals, width=w, label="ViT", color=series_color("vit"))
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(orgs)
    ax.set_title("Per-Organ Mean Correlation: ViS vs ViT")
    ax.set_ylabel("Mean gene-wise r")
    ax.grid(False)
    annotate_vertical_bars(ax, decimals=3)
    set_legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_12_per_organ_grouped_bars.png")
    plt.close(fig)

    # 13) Per-organ delta (ViT - ViS).
    delta = np.array(vit_vals) - np.array(vis_vals)
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.bar(orgs, delta, width=0.50, color=["#2A9D8F" if d >= 0 else "#E76F51" for d in delta])
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_title("Per-Organ Delta: ViT - ViS (Mean r)")
    ax.set_ylabel("Delta mean r")
    ax.grid(False)
    annotate_vertical_bars(ax, decimals=3, y_offset_frac=0.015)
    plt.tight_layout()
    fig.savefig(OUT / "fig_13_per_organ_delta_vit_minus_vis.png")
    plt.close(fig)

    # 14) Slide-level correlation distributions.
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(slide_comp[["meanpool_mean_corr", "vit_mean_corr"]].min()), max(slide_comp[["meanpool_mean_corr", "vit_mean_corr"]].max()), 28)
    ax.hist(slide_comp["meanpool_mean_corr"].dropna(), bins=bins, alpha=0.65, label="Meanpool", color="#9C6644")
    ax.hist(slide_comp["vit_mean_corr"].dropna(), bins=bins, alpha=0.65, label="ViT", color="#F07167")
    ax.set_title("Held-out Slide: Distribution of Mean Correlation")
    ax.set_xlabel("Mean gene-wise r per held-out slide")
    ax.set_ylabel("Count of slides")
    ax.grid(False)
    ax.axvline(slide_comp["meanpool_mean_corr"].mean(), color="#9C6644", linestyle="--", linewidth=1.3, label="Meanpool mean")
    ax.axvline(slide_comp["vit_mean_corr"].mean(), color="#F07167", linestyle="--", linewidth=1.3, label="ViT mean")
    set_legend_above(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_14_slide_corr_hist.png")
    plt.close(fig)

    # 15) Slide-level model comparison scatter.
    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.scatter(slide_comp["meanpool_mean_corr"], slide_comp["vit_mean_corr"], s=70, color="#4D908E", alpha=0.85)
    lims = [
        min(slide_comp["meanpool_mean_corr"].min(), slide_comp["vit_mean_corr"].min()) - 0.01,
        max(slide_comp["meanpool_mean_corr"].max(), slide_comp["vit_mean_corr"].max()) + 0.01,
    ]
    ax.plot(lims, lims, "--", color="#555")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Meanpool mean r")
    ax.set_ylabel("ViT mean r")
    ax.set_title("Held-out Slide: ViT vs Meanpool")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(OUT / "fig_15_slide_scatter_vit_vs_meanpool.png")
    plt.close(fig)

    # 16) Top/bottom slides by ViT.
    vit_rank = slide_comp.sort_values("vit_mean_corr", ascending=False)
    top = vit_rank.head(6)
    bot = vit_rank.tail(6)
    rank_df = pd.concat([top, bot], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2A9D8F"] * len(top) + ["#E76F51"] * len(bot)
    ax.barh(rank_df["heldout_slide"], rank_df["vit_mean_corr"], color=colors)
    ax.axvline(0, color="#333", linewidth=1)
    ax.set_title("Best and Worst Held-out Slides by ViT")
    ax.set_xlabel("Mean gene-wise r")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(OUT / "fig_16_slide_top_bottom_vit.png")
    plt.close(fig)

    # 17) Run status counts.
    status_counts = run_status.groupby("status", as_index=False).size().rename(columns={"size": "count"})
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    color_map = {"completed": "#2A9D8F", "failed": "#E76F51", "timeout": "#F4A261", "oom": "#B56576"}
    ax.bar(status_counts["status"], status_counts["count"], width=0.50, color=[color_map.get(s, "#888") for s in status_counts["status"]])
    ax.set_title("Experiment Run Status Overview")
    ax.set_ylabel("Count")
    ax.grid(False)
    annotate_vertical_bars(ax, decimals=0)
    plt.tight_layout()
    fig.savefig(OUT / "fig_17_run_status_counts.png")
    plt.close(fig)

    # 18) ID leaderboard horizontal plot.
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = phase2_ext.sort_values("mean_gene_correlation", ascending=True)
    ax.barh([pretty_label(m) for m in ordered["model"]], ordered["mean_gene_correlation"], color=[series_color(m) for m in ordered["model"]])
    ax.set_title("ID Leaderboard (Mean Correlation)")
    ax.set_xlabel("Mean gene-wise r")
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(OUT / "fig_18_id_leaderboard_horizontal.png")
    plt.close(fig)

    print(f"Presentation assets written to: {OUT}")


if __name__ == "__main__":
    main()
