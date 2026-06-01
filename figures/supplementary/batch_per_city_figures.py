"""Regenerate three SI figures per city from the local regmodels artifacts.

Produces, for every city listed in ``CITIES``:

- ``S_<country>_<city>_fold.pdf``      -- per-indicator 5-fold R^2 (box+strip)
- ``S_<country>_<city>_random.pdf``    -- R^2 vs sampling ratio under random sampling
                                           (mean +- std over 10 folds)
- ``S_<country>_<city>_strategic.pdf`` -- R^2 vs sampling ratio under strategic sampling

The 5-fold figure reads the per-fold predictions in ``Fold/<C>/<city>/Fuse/Multi_Concat/results.h5``
(keys F0..F4, each a DataFrame with columns ``target`` + ``pred_<target>`` + ``set``),
computing the test-set R^2 per target per fold. The two sampling figures read the
precomputed ``Ratio/.../results.csv`` (10 folds per ratio) and
``Sampling/.../Multi_Concat_pcahierachy/results.csv`` (single run per ratio).

Target -> display label + SDG category mapping is hardcoded in ``META`` below,
derived from the SI tables. Non-AU/US mappings are best-effort and may need
adjustment by the authors.
"""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_ROOT = "../../data/regression_outputs/regmodels"
OUT_DIR = "../../docs/figure_outputs"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "../../data/Arial.ttf",
]
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = "Arial"
        break
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Official SDG colour palette.
SDG_COLORS = {
    1: "#e5243b",  3: "#4C9F38",  4: "#C5192D",  5: "#FF3A21",
    6: "#26BDE2",  8: "#A21942",  9: "#FD6925", 10: "#DD1367",
    11: "#FD9D24", 13: "#3F7E44", 16: "#00689D",
}

# target -> (display Code, SDG)
# Confident mappings (AU, US) come from the SI tables and processed/0labels;
# BR/CN/FR/PT/NG mappings are best-effort given the raw variable definitions in
# the legacy 1preprocess_country_raw/<country>.ipynb notebooks and the SI tables.
META: dict[str, dict[str, Tuple[str, int]]] = {
    "Australia": {
        "med_hhinc":      ("Income",              1),
        "age65":          ("Age Index",           3),
        "arthritis":      ("% Arthritis",         3),
        "asthma":         ("% Asthma",            3),
        "cancer":         ("% Cancer",            3),
        "diabetes":       ("% Diabetes",          3),
        "heart_disease":  ("% Heart",             3),
        "kidney_disease": ("% Kidney",            3),
        "lung_condition": ("% Lung",              3),
        "mental_health":  ("% Mental Health",     3),
        "edu_year12":     ("% Edu complete",      4),
        "edu_noschool":   ("% Edu drop",          4),
        "med_capgain":    ("Median Capital",      8),
        "mean_capgain":   ("Mean Capital",        8),
        "unemploy":       ("% Unemploy",          8),
        "internet":       ("% Internet",          9),
        "popden":         ("Pop Density",        11),
        "renter":         ("Renter",             11),
        "publictrans":    ("% Public Transit",   11),
        "drive":          ("% Drive",            11),
        "bike":           ("% Bicycle",          11),
        "walk":           ("% Walk",             11),
    },
    "US": {
        "logincome":             ("Income",                    1),
        "povertyline_below100":  ("% Poverty Line (100%)",     1),
        "povertyline_below200":  ("% Poverty Line (200%)",     1),
        "cancercrud":            ("% Cancer Health",           3),
        "diabetescr":            ("% Diabetes",                3),
        "obesitycru":            ("% Obesity",                 3),
        "lpacrudepr":            ("% LPA",                     3),
        "mhlthcrude":            ("% Mental Health",           3),
        "phlthcrude":            ("% Physical Health",         3),
        "walkbike_per_cbg":      ("% Walk",                   13),
        "publictrans_per_cbg":   ("% Public Transit",         13),
        "drove_alone_per_cbg":   ("% Drive Alone",            13),
        "estvmiles":             ("VMT",                      13),
        "estpmiles":             ("PMT",                      13),
        "estvtrp":               ("VTRP",                     13),
        "estptrp":               ("PTRP",                     13),
        "logcrime":              ("% Violent Crime",          16),
        "logpetty":              ("% Petty Crime",            16),
    },
    # NOTE: best-effort mappings below; please verify against 0labels/*.csv.
    "Brazil": {
        "BR01": ("Pop Density",       11),
        "BR02": ("% Elderly",          3),
        "BR12": ("% Adult Literacy",   4),
        "BR13": ("% Gender",           5),
        "BR14": ("Avg Residence",     11),
        "BR17": ("Color Diversity",   10),
    },
    "China": {
        "HK02": ("Income",             1),
        "HK06": ("Age Index",          3),
        "HK07": ("% Uneducated",       4),
        "HK15": ("% Labor",            8),
        "HK17": ("% Chinese",         16),
    },
    "France": {
        "FR02": ("Age Index",          3),
        "FR03": ("% Employ (Women)",   8),
        "FR04": ("% Employ (Men)",     8),
        "FR08": ("Gini Index",        10),
        "FR09": ("% Overcrowded",     11),
        "FR11": ("Pop Density",       11),
    },
    "Portugal": {
        "PT16": ("% Uneducated",       4),
        "PT24": ("% High Education",   4),
        "PT28": ("% Activity",         8),
        "PT33": ("% Unemployment",     8),
        "PT34": ("% Disable",         10),
        "PT35": ("% Wheelchair",      11),
        "PT37": ("Pop Density",       11),
        "PT38": ("Renter",            11),
    },
    "Nigeria": {
        "pub_health": ("Pub Health Den", 3),
        "health_den": ("Health Den",     6),
        "market_den": ("Market Den",     8),
    },
}

CITIES = [
    ("Australia", "Adelaide"),    ("Australia", "Brisbane"),
    ("Australia", "Melbourne"),   ("Australia", "Perth"),
    ("Australia", "Sydney"),
    ("Brazil", "BeloHorizonte"),  ("Brazil", "Curitiba"),
    ("Brazil", "PortoAlegre"),    ("Brazil", "RiodeJaneiro"),
    ("China", "HongKong"),
    ("France", "All"),
    ("Portugal", "All"),
    ("US", "Boston"),             ("US", "Chicago"),
    ("US", "LosAngeles"),         ("US", "Miami"),
    ("US", "NewYork"),            ("US", "Philadelphia"),
    ("US", "SanFrancisco"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _annotate(df: pd.DataFrame, country: str) -> pd.DataFrame:
    meta = META.get(country, {})
    df = df.copy()
    df["Code"] = df["target"].map(lambda t: meta.get(t, (t, 0))[0])
    df["SDG"] = df["target"].map(lambda t: meta.get(t, (t, 0))[1])
    return df.sort_values(["SDG", "Code"]).reset_index(drop=True)


def load_fold_r2(country: str, city: str) -> pd.DataFrame | None:
    """Return long df [target, Code, SDG, fold, r2] from Fold/.../results.h5."""
    p = f"{MODEL_ROOT}/Fold/{country}/{city}/Fuse/Multi_Concat/results.h5"
    if not os.path.exists(p):
        return None
    records = []
    for k in ["F0", "F1", "F2", "F3", "F4"]:
        try:
            df = pd.read_hdf(p, key=k)
        except KeyError:
            continue
        test = df[df["set"] == "test"] if "set" in df.columns else df
        targets = [c[5:] for c in test.columns if c.startswith("pred_")]
        for t in targets:
            if t not in test.columns:
                continue
            m = test[t].notna() & test[f"pred_{t}"].notna()
            if m.sum() < 3:
                continue
            r2 = r2_score(test.loc[m, t], test.loc[m, f"pred_{t}"])
            records.append({"target": t, "fold": int(k[1:]), "r2": r2})
    if not records:
        return None
    return _annotate(pd.DataFrame(records), country)


def load_ratio(country: str, city: str) -> pd.DataFrame | None:
    p = f"{MODEL_ROOT}/Ratio/{country}/{city}/Fuse/Multi_Concat/results.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if {"target", "ratio", "fold", "R2"} - set(df.columns):
        return None
    return _annotate(df[["target", "ratio", "fold", "R2"]].rename(columns={"R2": "r2"}), country)


def load_strategic(country: str, city: str) -> pd.DataFrame | None:
    # Folder name varies (Multi_Concat_pcahierachy or similar); glob for any child.
    import glob
    hits = glob.glob(f"{MODEL_ROOT}/Sampling/{country}/{city}/Fuse/Multi_Concat*/results.csv")
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    if {"target", "ratio", "R2"} - set(df.columns):
        return None
    return _annotate(df[["target", "ratio", "R2"]].rename(columns={"R2": "r2"}), country)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", labelsize=7, color="black")
    ax.axhline(0, color="lightgray", linewidth=0.5, linestyle="--", zorder=0)


def _palette(codes: list[str], code_to_sdg: dict[str, int]) -> dict[str, str]:
    return {c: SDG_COLORS.get(code_to_sdg.get(c, 0), "#808080") for c in codes}


def _legend(ax, sdgs: list[int]):
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=SDG_COLORS[s], label=f"SDG{s}")
        for s in sdgs if s in SDG_COLORS
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, loc="best", fontsize=7)


def plot_fold(country: str, city: str):
    df = load_fold_r2(country, city)
    if df is None or df.empty:
        print(f"[skip fold]       {country}/{city}")
        return
    order = df.drop_duplicates("Code").sort_values(["SDG", "Code"])["Code"].tolist()
    c2s = dict(zip(df["Code"], df["SDG"]))
    pal = _palette(order, c2s)

    fig, ax = plt.subplots(figsize=(18 / 2.5, 8 / 2.5))
    sns.boxplot(data=df, x="Code", y="r2", hue="Code", order=order, palette=pal,
                width=0.55, linewidth=0.8, fliersize=0, legend=False, ax=ax)
    sns.stripplot(data=df, x="Code", y="r2", hue="Code", order=order, palette=pal,
                  size=2.8, alpha=0.85, jitter=0.15, edgecolor="white",
                  linewidth=0.3, legend=False, ax=ax)
    _style(ax)
    ax.set_ylabel(r"$R^2$", fontsize=8)
    ax.set_xlabel("", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    _legend(ax, sorted({c2s[c] for c in order}))

    out = f"{OUT_DIR}/S_{country}_{city}_fold.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok  fold]        {out}")


def _plot_curves(df: pd.DataFrame, title: str, out: str, has_folds: bool):
    order = df.drop_duplicates("Code").sort_values(["SDG", "Code"])["Code"].tolist()
    c2s = dict(zip(df["Code"], df["SDG"]))
    pal = _palette(order, c2s)

    fig, ax = plt.subplots(figsize=(18 / 2.5, 8 / 2.5))
    for code in order:
        sub = df[df["Code"] == code].sort_values("ratio")
        if has_folds:
            grp = sub.groupby("ratio")["r2"].agg(["mean", "std"]).reset_index()
            ax.plot(grp["ratio"], grp["mean"], color=pal[code], linewidth=1.1,
                    marker="o", markersize=3)
            ax.fill_between(grp["ratio"], grp["mean"] - grp["std"],
                            grp["mean"] + grp["std"], color=pal[code], alpha=0.18,
                            linewidth=0)
        else:
            ax.plot(sub["ratio"], sub["r2"], color=pal[code], linewidth=1.1,
                    marker="o", markersize=3)

    _style(ax)
    ax.set_xlabel("Sampling ratio", fontsize=8)
    ax.set_ylabel(r"Overall $R^2$", fontsize=8)
    ax.set_xlim(0.05, 0.95)
    ax.axhline(0.8, color="gray", linewidth=0.6, linestyle=":", zorder=0)
    _legend(ax, sorted({c2s[c] for c in order}))

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok  {title:<9}] {out}")


def plot_random(country: str, city: str):
    df = load_ratio(country, city)
    if df is None or df.empty:
        print(f"[skip random]     {country}/{city}")
        return
    out = f"{OUT_DIR}/S_{country}_{city}_random.pdf"
    _plot_curves(df, "random", out, has_folds=True)


def plot_strategic(country: str, city: str):
    df = load_strategic(country, city)
    if df is None or df.empty:
        print(f"[skip strategic]  {country}/{city}")
        return
    out = f"{OUT_DIR}/S_{country}_{city}_strategic.pdf"
    _plot_curves(df, "strategic", out, has_folds=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for country, city in CITIES:
        plot_fold(country, city)
        plot_random(country, city)
        plot_strategic(country, city)


if __name__ == "__main__":
    main()
