#!/usr/bin/env python3
"""Two figures that are not produced by pdac_pipeline.py.

fig_permutation_null.png  observed balanced accuracy vs the 20-shuffle null,
                          per method, for the three analytes that have a null
fig_seed_spread.png       five-seed range of balanced accuracy for Step 2 and
                          Step 3 on every analyte, published value marked

Usage:  python scripts/make_extra_figs.py [results/loo_5seeds/report]
Both figures are written next to the CSVs they are built from.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/loo_5seeds/report")

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
NULL_GREY = "#b5b3ab"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})

PRETTY = {
    "Tissue_Protein": "Tissue Protein", "Plasma_Protein": "Plasma Protein",
    "Plasma_Lipid": "Plasma Lipid", "RNA_Fusion": "RNA Fusion",
    "Multiomic": "Multi-omic",
}
METHOD_LABEL = {
    "BASE_AUTO": "Step 2 (released, nested)", "PIPE_AUTO": "Step 3 (P1–P6, nested)",
    "P1": "P1 L1+SMOTE+RF", "P2": "P2 RFECV+XGBoost", "P4": "P4 t-test+SMOTE+RF",
    "P5": "P5 L1+NearMiss+RF", "P6": "P6 LightGBM DART",
    "SVM": "SVM", "L1_SVM": "L1 SVM", "L1_RF": "L1 RF", "PCA_LR": "PCA LR",
    "DNN": "L1 MLP", "RFE_LR": "RFECV LR", "RFE_RF": "RFECV RF",
}
METHOD_ORDER = ["PIPE_AUTO", "P1", "P5", "P4", "P2", "P6",
                "BASE_AUTO", "SVM", "L1_SVM", "L1_RF", "PCA_LR", "DNN", "RFE_LR", "RFE_RF"]
ANALYTE_ORDER = ["Clinical", "SNV", "INDEL", "CNV", "RNA", "RNA_Fusion", "Pathology",
                 "Plasma_Protein", "Plasma_Lipid", "Tissue_Protein", "Multiomic"]


def pretty(a):
    return PRETTY.get(a, a)


# --------------------------------------------------------------------------
# Figure 1: permutation null
# --------------------------------------------------------------------------
def fig_permutation_null(ml, null_check):
    perm = ml[ml["labels"].str.startswith("perm_")]
    obs = ml[(ml["labels"] == "moesm3") & (ml["seed"] == 42)]
    analytes = [a for a in ["Tissue_Protein", "Plasma_Protein", "Multiomic"]
                if a in set(perm["Analyte"])]
    fig, axes = plt.subplots(1, len(analytes), figsize=(9.6, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(METHOD_ORDER))[::-1]

    for ax, analyte in zip(axes, analytes):
        for yi, m in zip(y, METHOD_ORDER):
            nulls = perm[(perm["Analyte"] == analyte) & (perm["Method"] == m)]["BA"].values
            o = obs[(obs["Analyte"] == analyte) & (obs["Method"] == m)]["BA"]
            if len(nulls) == 0 or o.empty:
                continue
            o = float(o.iloc[0])
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-0.18, 0.18, size=len(nulls))
            ax.scatter(nulls, yi + jitter, s=9, color=NULL_GREY, alpha=0.8, lw=0, zorder=2)
            ax.plot([nulls.max()] * 2, [yi - 0.3, yi + 0.3], color=INK2, lw=1, zorder=3)
            beats_all = o > nulls.max()
            ax.scatter([o], [yi], s=46, marker="D" if beats_all else "o",
                       color=BLUE if beats_all else ORANGE, edgecolor="white",
                       lw=0.8, zorder=4)
            if beats_all:
                ax.text(o + 0.025, yi, f"{o:.2f}", va="center", ha="left",
                        fontsize=7.5, color=INK)
        ax.axvline(0.5, color=GRID, lw=1, zorder=1)
        ax.axhline(y[6] + 0.5, color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(0.25, 0.95)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        k = perm[perm["Analyte"] == analyte]["labels"].nunique()
        ax.set_title(f"{pretty(analyte)}  ({k} shuffles)")
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Balanced accuracy (leave-one-out)")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])

    h = [
        plt.Line2D([], [], marker="o", color=NULL_GREY, ls="", ms=4,
                   label="one label shuffle"),
        plt.Line2D([], [], color=INK2, lw=1, label="best shuffle"),
        plt.Line2D([], [], marker="D", color=BLUE, ls="", ms=6,
                   label="observed, beats every shuffle"),
        plt.Line2D([], [], marker="o", color=ORANGE, ls="", ms=6,
                   label="observed, within the null"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.suptitle("Observed balanced accuracy against the permutation null "
                 "(labels shuffled, class counts kept)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = REPORT / "fig_permutation_null.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Figure 2: seed spread
# --------------------------------------------------------------------------
def fig_seed_spread(ml, t1):
    d = ml[(ml["labels"] == "moesm3") & ml["Method"].isin(["BASE_AUTO", "PIPE_AUTO"])]
    pub = t1.set_index("Analyte")["Published_BA"]
    analytes = [a for a in ANALYTE_ORDER if a in set(d["Analyte"])]
    y = np.arange(len(analytes))[::-1]
    off = {"BASE_AUTO": +0.18, "PIPE_AUTO": -0.18}
    col = {"BASE_AUTO": ORANGE, "PIPE_AUTO": BLUE}

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for yi, a in zip(y, analytes):
        for m in ["BASE_AUTO", "PIPE_AUTO"]:
            v = d[(d["Analyte"] == a) & (d["Method"] == m)]["BA"].values
            if len(v) == 0:
                continue
            yy = yi + off[m]
            ax.plot([v.min(), v.max()], [yy, yy], color=col[m], lw=2.2,
                    solid_capstyle="round", zorder=2)
            ax.scatter(v, [yy] * len(v), s=12, color=col[m], edgecolor="white",
                       lw=0.6, zorder=3)
            ax.scatter([v.mean()], [yy], s=42, color=col[m], edgecolor="white",
                       lw=0.9, zorder=4)
        ax.scatter([pub[a]], [yi], s=70, marker="|", color=INK, lw=1.8, zorder=5)
    ax.axvline(0.5, color=GRID, lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(a) for a in analytes])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.3, 0.9)
    ax.set_xlabel("Balanced accuracy (leave-one-out), range and mean over seeds 42–46")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color=GRID, lw=0.5, zorder=0)

    h = [
        plt.Line2D([], [], color=ORANGE, lw=2.2, marker="o", ms=6, label="Step 2, released models (nested)"),
        plt.Line2D([], [], color=BLUE, lw=2.2, marker="o", ms=6, label="Step 3, pipelines P1–P6 (nested)"),
        plt.Line2D([], [], color=INK, ls="", marker="|", ms=9, mew=1.8,
                   label="published Table 1, as balanced accuracy"),
    ]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, frameon=False, fontsize=8)
    ax.set_title("Seed sensitivity: five-seed range per analyte")
    fig.tight_layout()
    out = REPORT / "fig_seed_spread.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ml = pd.read_csv(REPORT / "metrics_long.csv")
    null_check = pd.read_csv(REPORT / "null_check.csv")
    t1 = pd.read_csv(REPORT / "table1_dataset.csv")
    print("wrote", fig_permutation_null(ml, null_check))
    print("wrote", fig_seed_spread(ml, t1))
