"""
Bar charts showing R² for each individual indicator in 4 representative cities.
Layout: 2x2 panels, one per city, horizontal bars colored by SDG category.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------
FONT_PATH = "C:/Windows/Fonts/arial.ttf"
font_prop = fm.FontProperties(fname=FONT_PATH)
font_prop_bold = fm.FontProperties(fname=FONT_PATH, weight="bold")
plt.rcParams.update({
    "font.family": font_prop.get_name(),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

# ---------------------------------------------------------------------------
# SDG colors
# ---------------------------------------------------------------------------
SDG_COLORS = {
    "SDG1": "#E5243B",
    "SDG3": "#4C9F38",
    "SDG4": "#C5192D",
    "SDG5": "#FF3A21",
    "SDG6": "#26BDE2",
    "SDG8": "#A21942",
    "SDG9": "#FD6925",
    "SDG10": "#DD1367",
    "SDG11": "#FD9D24",
    "SDG13": "#3F7E44",
    "SDG16": "#00689D",
}

# ---------------------------------------------------------------------------
# Indicator -> SDG mapping per city
# ---------------------------------------------------------------------------
SDG_MAP = {
    "Philadelphia": {
        "logincome": "SDG1", "povertyline_below100": "SDG1", "povertyline_below200": "SDG1",
        "cancercrud": "SDG3", "diabetescr": "SDG3", "obesitycru": "SDG3",
        "lpacrudepr": "SDG3", "mhlthcrude": "SDG3", "phlthcrude": "SDG3",
        "drove_alone_per_cbg": "SDG13", "publictrans_per_cbg": "SDG13",
        "walkbike_per_cbg": "SDG13", "estpmiles": "SDG13", "estptrp": "SDG13",
        "estvmiles": "SDG13", "estvtrp": "SDG13",
        "logcrime": "SDG16", "logpetty": "SDG16",
    },
    "Melbourne": {
        "med_hhinc": "SDG1",
        "age65": "SDG3", "arthritis": "SDG3", "asthma": "SDG3", "cancer": "SDG3",
        "diabetes": "SDG3", "heart_disease": "SDG3", "kidney_disease": "SDG3",
        "lung_condition": "SDG3", "mental_health": "SDG3",
        "edu_year12": "SDG4", "edu_noschool": "SDG4",
        "mean_capgain": "SDG8", "med_capgain": "SDG8", "unemploy": "SDG8",
        "internet": "SDG9",
        "popden": "SDG11", "renter": "SDG11",
        "drive": "SDG13", "publictrans": "SDG13", "walk": "SDG13", "bike": "SDG13",
    },
    "RiodeJaneiro": {
        "BR01": "SDG1", "BR02": "SDG3", "BR12": "SDG4",
        "BR13": "SDG5", "BR14": "SDG10", "BR17": "SDG11",
    },
    "HongKong": {
        "HK02": "SDG1", "HK06": "SDG4", "HK07": "SDG8",
        "HK15": "SDG10", "HK17": "SDG16",
    },
}

# ---------------------------------------------------------------------------
# Readable indicator names
# ---------------------------------------------------------------------------
INDICATOR_LABELS = {
    # US - Philadelphia
    "logincome": "Income (log)",
    "povertyline_below100": "Poverty <100%",
    "povertyline_below200": "Poverty <200%",
    "cancercrud": "Cancer",
    "diabetescr": "Diabetes",
    "obesitycru": "Obesity",
    "lpacrudepr": "Physical inactivity",
    "mhlthcrude": "Poor mental health",
    "phlthcrude": "Poor physical health",
    "drove_alone_per_cbg": "Drove alone",
    "publictrans_per_cbg": "Public transit",
    "walkbike_per_cbg": "Walk/bike",
    "estpmiles": "Person-miles",
    "estptrp": "Person-trips",
    "estvmiles": "Vehicle-miles",
    "estvtrp": "Vehicle-trips",
    "logcrime": "Crime (log)",
    "logpetty": "Petty crime (log)",
    # Australia - Melbourne
    "med_hhinc": "Median income",
    "age65": "Age 65+",
    "arthritis": "Arthritis",
    "asthma": "Asthma",
    "cancer": "Cancer",
    "diabetes": "Diabetes",
    "heart_disease": "Heart disease",
    "kidney_disease": "Kidney disease",
    "lung_condition": "Lung condition",
    "mental_health": "Mental health",
    "edu_year12": "Year 12 education",
    "edu_noschool": "No schooling",
    "mean_capgain": "Mean capital gain",
    "med_capgain": "Median capital gain",
    "unemploy": "Unemployment",
    "internet": "Internet access",
    "popden": "Population density",
    "renter": "Renters",
    "drive": "Drove",
    "publictrans": "Public transit",
    "walk": "Walked",
    "bike": "Cycled",
    # Brazil - Rio de Janeiro
    "BR01": "Income (SDG 1)",
    "BR02": "Health (SDG 3)",
    "BR12": "Education (SDG 4)",
    "BR13": "Gender (SDG 5)",
    "BR14": "Inequality (SDG 10)",
    "BR17": "Urbanization (SDG 11)",
    # China - Hong Kong
    "HK02": "Income (SDG 1)",
    "HK06": "Education (SDG 4)",
    "HK07": "Employment (SDG 8)",
    "HK15": "Inequality (SDG 10)",
    "HK17": "Safety (SDG 16)",
}

# ---------------------------------------------------------------------------
# City configs
# ---------------------------------------------------------------------------
CITIES = [
    {"name": "Philadelphia", "country": "US", "flag": "us.png", "label": "Philadelphia, US"},
    {"name": "Melbourne", "country": "Australia", "flag": "au.png", "label": "Melbourne, Australia"},
    {"name": "RiodeJaneiro", "country": "Brazil", "flag": "br.png", "label": "Rio de Janeiro, Brazil"},
    {"name": "HongKong", "country": "China", "flag": "cn.png", "label": "Hong Kong, China"},
]

DATA_ROOT = Path("../../data/regression_outputs/regmodels/Fold")
FLAG_DIR = Path("D:/Workspace/01Project/00_SDG/images/src/flags")
OUT_DIR = Path("D:/Workspace/01Project/00_SDG/images/extended")


def load_city_data(country, city):
    csv_path = DATA_ROOT / country / city / "Fuse" / "Multi_Concat" / "results.csv"
    df = pd.read_csv(csv_path)
    # Average across folds if multiple exist
    df = df.groupby("target", as_index=False)["r2"].mean()
    return df


def get_flag_image(flag_file, zoom=0.06):
    img = Image.open(FLAG_DIR / flag_file)
    return OffsetImage(np.array(img), zoom=zoom)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Collect all SDGs used for the legend
    all_sdgs_used = set()

    for idx, city_cfg in enumerate(CITIES):
        ax = axes[idx]
        city = city_cfg["name"]
        country = city_cfg["country"]
        sdg_map = SDG_MAP[city]

        df = load_city_data(country, city)
        df["sdg"] = df["target"].map(sdg_map)
        df["label"] = df["target"].map(INDICATOR_LABELS).fillna(df["target"])
        df["color"] = df["sdg"].map(SDG_COLORS)
        df = df.sort_values("r2", ascending=True).reset_index(drop=True)

        all_sdgs_used.update(df["sdg"].dropna().unique())

        y_pos = np.arange(len(df))
        bars = ax.barh(
            y_pos, df["r2"], height=0.7,
            color=df["color"].values, edgecolor="white", linewidth=0.3,
            zorder=3,
        )

        # Value labels on bars
        for i, (val, bar) in enumerate(zip(df["r2"], bars)):
            if val >= 0:
                ax.text(
                    val + 0.01, i, f"{val:.2f}",
                    va="center", ha="left", fontsize=7,
                    fontproperties=font_prop, color="#333333",
                )
            else:
                ax.text(
                    val - 0.01, i, f"{val:.2f}",
                    va="center", ha="right", fontsize=7,
                    fontproperties=font_prop, color="#333333",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["label"], fontproperties=font_prop, fontsize=8)
        ax.set_xlabel("R²", fontproperties=font_prop_bold, fontsize=10)

        # X-axis limits
        x_min = min(0, df["r2"].min() - 0.05)
        ax.set_xlim(x_min, 1.05)

        # Add vertical line at 0 if there are negative values
        if df["r2"].min() < 0:
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="-", zorder=2)

        # Grid
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
        ax.yaxis.grid(False)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0)

        # Panel label (a, b, c, d) and city name with flag
        panel_letter = chr(ord("a") + idx)

        # Add flag icon after title text
        try:
            flag_img = get_flag_image(city_cfg["flag"], zoom=0.04)
            title_text = f"({panel_letter}) {city_cfg['label']}"
            t = ax.set_title(title_text, fontproperties=font_prop_bold, fontsize=11,
                             loc="left", pad=10)
            # Place flag to the right of the title using figure-level positioning
            # We use a text-relative approach: annotate after title
            fig.canvas.draw()
            bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = ax.transAxes.inverted()
            title_end_x = inv.transform(bbox)[1][0]  # right edge in axes coords
            ab = AnnotationBbox(
                flag_img,
                (title_end_x + 0.02, 1.04), xycoords="axes fraction",
                box_alignment=(0, 0.5),
                frameon=False,
                pad=0,
            )
            ax.add_artist(ab)
        except Exception:
            ax.set_title(f"({panel_letter}) {city_cfg['label']}",
                         fontproperties=font_prop_bold, fontsize=11, loc="left", pad=8)

    # ---------------------------------------------------------------------------
    # SDG Legend
    # ---------------------------------------------------------------------------
    sdg_order = sorted(all_sdgs_used, key=lambda s: int(s.replace("SDG", "")))
    sdg_full_names = {
        "SDG1": "SDG 1: No Poverty",
        "SDG3": "SDG 3: Good Health",
        "SDG4": "SDG 4: Quality Education",
        "SDG5": "SDG 5: Gender Equality",
        "SDG8": "SDG 8: Decent Work",
        "SDG9": "SDG 9: Industry & Innovation",
        "SDG10": "SDG 10: Reduced Inequalities",
        "SDG11": "SDG 11: Sustainable Cities",
        "SDG13": "SDG 13: Climate Action",
        "SDG16": "SDG 16: Peace & Justice",
    }
    legend_handles = []
    for sdg in sdg_order:
        patch = FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.1",
            facecolor=SDG_COLORS[sdg],
            edgecolor="none",
        )
        legend_handles.append(patch)

    legend_labels = [sdg_full_names.get(s, s) for s in sdg_order]
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        prop=font_prop,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"indicator_bars.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")

    plt.close()


if __name__ == "__main__":
    main()
