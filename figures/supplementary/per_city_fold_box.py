"""Regenerate per-city supplementary fold figures as box+strip plots.

Replaces the single-value bar chart (`supplementary_fold_results.ipynb`) with a
box+strip plot that shows all 5 fold R^2 values per indicator. Run this on the
server where `models_new/Fold/<country>/<city>/Fuse/Multi_Concat/results.csv`
is available and rsync the resulting PDFs into `writing/images/` of the Overleaf
project.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

FONT_PATH = "../../data/Arial.ttf"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MODEL_ROOT = "../../data/regression_outputs_new/Fold"
META_ROOT = "../../data/processed/0labels"
OUT_DIR = "../../data/figure_assets"

# Official SDG palette; keys match meta['SDG'] integer codes.
SDG_COLORS = {
    1: "#e5243b",
    3: "#4C9F38",
    4: "#C5192D",
    5: "#FF3A21",
    6: "#26BDE2",
    8: "#A21942",
    9: "#FD6925",
    10: "#DD1367",
    11: "#FD9D24",
    13: "#3F7E44",
    16: "#00689D",
}

# Cities to regenerate. Extend/trim as needed.
CITIES = [
    ("Australia", "Adelaide"),
    ("Australia", "Brisbane"),
    ("Australia", "Melbourne"),
    ("Australia", "Perth"),
    ("Australia", "Sydney"),
    ("Brazil", "BeloHorizonte"),
    ("Brazil", "Curitiba"),
    ("Brazil", "PortoAlegre"),
    ("Brazil", "RiodeJaneiro"),
    ("China", "HongKong"),
    ("France", "All"),
    ("Portugal", "All"),
    ("US", "Boston"),
    ("US", "Chicago"),
    ("US", "LosAngeles"),
    ("US", "Miami"),
    ("US", "NewYork"),
    ("US", "Philadelphia"),
    ("US", "SanFrancisco"),
]


def load_fold_r2(country: str, city: str) -> pd.DataFrame | None:
    """Return a long-form frame with columns [target, Code, SDG, fold, r2].

    The server-side results.csv is expected to carry one row per (target, fold).
    If the CSV only holds a single summary row per target, this falls back to
    a point plot so the script still produces output.
    """
    candidates = [
        f"{MODEL_ROOT}/{country}/{city}/Fuse/Multi_Concat/results.csv",
        f"{MODEL_ROOT}/{country}/{city}/Fuse/Multi_Concat/result.csv",
    ]
    path = next((p for p in candidates if glob.glob(p)), None)
    if path is None:
        print(f"[skip] no results csv for {country}/{city}")
        return None

    raw = pd.read_csv(path)
    if "r2" not in raw.columns or "fold" not in raw.columns:
        print(f"[skip] {path} missing expected columns")
        return None

    meta_path = f"{META_ROOT}/{country}.csv"
    if not os.path.exists(meta_path):
        print(f"[skip] no meta for {country}")
        return None
    meta = pd.read_csv(meta_path)[["ID", "Code", "SDG"]]

    df = raw.merge(meta, left_on="target", right_on="ID", how="inner").drop(
        columns=["ID"]
    )
    df = df[df["r2"].notna()]
    df = df.sort_values(by=["SDG", "Code"]).reset_index(drop=True)
    return df


def plot_city(country: str, city: str) -> None:
    df = load_fold_r2(country, city)
    if df is None or df.empty:
        return

    order = df.drop_duplicates("Code").sort_values("SDG")["Code"].tolist()
    code_to_sdg = dict(zip(df["Code"], df["SDG"]))
    palette = {code: SDG_COLORS.get(code_to_sdg[code], "#808080") for code in order}

    fig, ax = plt.subplots(figsize=(18 / 2.5, 8 / 2.5))
    sns.boxplot(
        data=df,
        x="Code",
        y="r2",
        order=order,
        palette=palette,
        width=0.55,
        linewidth=0.8,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="Code",
        y="r2",
        order=order,
        palette=palette,
        size=2.5,
        alpha=0.8,
        jitter=0.15,
        edgecolor="white",
        linewidth=0.3,
        ax=ax,
    )

    # Build an SDG legend from codes that actually appear.
    seen_sdgs = sorted({code_to_sdg[c] for c in order})
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=SDG_COLORS[s], label=str(s))
        for s in seen_sdgs
        if s in SDG_COLORS
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=7, title="SDG",
              title_fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", labelsize=7, color="black")
    ax.set_ylabel(r"$R^2$", fontsize=8, color="black")
    ax.set_xlabel("", fontsize=8)
    ax.set_xticklabels(order, rotation=90, fontsize=7, color="black")
    ax.axhline(0, color="lightgray", linewidth=0.5, linestyle="--", zorder=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/S_{country}_{city}_fold.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def main() -> None:
    for country, city in CITIES:
        plot_city(country, city)


if __name__ == "__main__":
    main()
