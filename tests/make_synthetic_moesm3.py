#!/usr/bin/env python3
"""
Write a synthetic spreadsheet with the same layout as MOESM3 (same column
prefixes, 74 patients, the same per-analyte patient counts, 39 complete).
It contains NO real data. Use it to check that the pipeline runs before
spending compute on the real file.

  python tests/make_synthetic_moesm3.py synthetic.xlsx             # pure noise
  python tests/make_synthetic_moesm3.py synthetic.xlsx --signal 2  # planted signal
"""
import argparse

import numpy as np
import pandas as pd

BLOCKS = {  # analyte -> (prefix, n_features, group)
    "Clinical": ("clinical_", 40, "clin"), "SNV": ("freebayes_SNV_", 60, "dna"),
    "INDEL": ("pindel_INDEL_", 25, "dna"), "CNV": ("CNV_", 60, "dna"),
    "RNA": ("rna_expr_", 120, "rna"), "RNA_Fusion": ("AF4_", 29, "rna"),
    "Pathology": ("pathology_NF", 50, "path"), "Plasma_Protein": ("plasma_protein_", 40, "plasma"),
    "Plasma_Lipid": ("plasma_lipid_", 40, "plasma"), "Tissue_Protein": ("tissue_protein_", 80, "tissue"),
}
GROUP_N = {"clin": 74, "dna": 72, "rna": 57, "path": 71, "plasma": 51, "tissue": 49}
N, N_COMPLETE, DEAD, DEAD_COMPLETE = 74, 39, 50, 28


def make(signal=0.0, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    while True:  # which of the 35 incomplete patients have each data group
        has = {g: np.r_[np.ones(N_COMPLETE, bool),
                        rng.permutation(np.r_[np.ones(n - N_COMPLETE, bool), np.zeros(N - n, bool)])]
               for g, n in GROUP_N.items()}
        if not np.all([has[g] for g in has], axis=0)[N_COMPLETE:].any():
            break
    y = np.r_[rng.permutation(np.r_[np.ones(DEAD_COMPLETE), np.zeros(N_COMPLETE - DEAD_COMPLETE)]),
              rng.permutation(np.r_[np.ones(DEAD - DEAD_COMPLETE),
                                    np.zeros(N - N_COMPLETE - (DEAD - DEAD_COMPLETE))])].astype(int)
    cols = {"Biobank_Number": [f"SYN{i:03d}" for i in range(N)], "label_deceased": y,
            "label_days_to_death": np.where(y == 1, rng.integers(60, 2500, N), np.nan),
            "label_recurrence": rng.integers(0, 2, N),      # extra outcome column: must not be used
            "notes_misc": rng.normal(size=N)}               # unassigned column: must not be used
    for name, (prefix, p, g) in BLOCKS.items():
        p = max(3, int(round(p * scale)))
        if g == "dna" or name == "RNA_Fusion":
            X = (rng.random((N, p)) < 0.15).astype(float)
        else:
            X = rng.normal(size=(N, p))
        k = max(1, p // 10)
        if signal:
            X[:, :k] += signal * (y[:, None] - 0.5) * (1 if X.dtype == float else 0)
        if g in ("plasma", "tissue"):                        # scattered missing values
            X[rng.random(X.shape) < 0.03] = np.nan
        X[~has[g]] = np.nan
        for j in range(p):
            cols[f"{prefix}{j:04d}"] = X[:, j]
    cols["clinical_days_to_last_followup"] = np.where(has["clin"], rng.integers(60, 2500, N), np.nan)
    sex = rng.choice(["F", "M"], N).astype(object)
    cols["clinical_sex_text"] = sex                           # non-numeric: audit should report it
    return pd.DataFrame(cols)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--signal", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply feature counts")
    a = ap.parse_args()
    df = make(a.signal, a.seed, a.scale)
    df.to_csv(a.out, index=False) if a.out.endswith(".csv") else df.to_excel(a.out, index=False)
    print(f"wrote {a.out}: {df.shape[0]} patients x {df.shape[1]} columns (synthetic)")
