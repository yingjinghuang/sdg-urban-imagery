"""
Plot small-multiple scaling curves (one panel per city) showing how
mean overall R² changes with sampling ratio, fitted with logarithmic curves.
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.optimize import curve_fit

# ── Paths ───────────────────────────────────────────────────────────────
DATA_ROOT = pathlib.Path("../../data/regression_outputs/regmodels/Ratio")
FLAG_DIR = pathlib.Path("D:/Workspace/01Project/00_SDG/images/src/flags")
OUT_DIR = pathlib.Path("D:/Workspace/01Project/00_SDG/images/extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/arial.ttf"
fm.fontManager.addfont(FONT_PATH)
FONT_PROP = fm.FontProperties(fname=FONT_PATH)
FONT_NAME = FONT_PROP.get_name()

# ── City definitions (country, folder_name, display_name, flag_code) ───
CITIES = [
    # Australia
    ("Australia", "Adelaide",       "Adelaide",        "au"),
    ("Australia", "Brisbane",       "Brisbane",        "au"),
    ("Australia", "Melbourne",      "Melbourne",       "au"),
    ("Australia", "Perth",          "Perth",           "au"),
    ("Australia", "Sydney",         "Sydney",          "au"),
    # Brazil
    ("Brazil", "BeloHorizonte",    "Belo Horizonte",  "br"),
    ("Brazil", "Curitiba",         "Curitiba",        "br"),
    ("Brazil", "PortoAlegre",      "Porto Alegre",    "br"),
    ("Brazil", "RiodeJaneiro",     "Rio de Janeiro",  "br"),
    # China
    ("China", "HongKong",          "Hong Kong",       "cn"),
    # France
    ("France", "All",              "France",          "fr"),
    # Nigeria
    ("Nigeria", "Lagos",           "Lagos",           "ng"),
    # Portugal
    ("Portugal", "All",            "Portugal",        "pt"),
    # US
    ("US", "Boston",               "Boston",          "us"),
    ("US", "Chicago",              "Chicago",         "us"),
    ("US", "LosAngeles",           "Los Angeles",     "us"),
    ("US", "Miami",                "Miami",           "us"),
    ("US", "NewYork",              "New York",        "us"),
    ("US", "Philadelphia",         "Philadelphia",    "us"),
    ("US", "SanFrancisco",         "San Francisco",   "us"),
]

COUNTRY_COLORS = {
    "China":     "#C0392B",
    "US":        "#2C3E50",
    "France":    "#5DADE2",
    "Brazil":    "#F1C40F",
    "Nigeria":   "#27AE60",
    "Portugal":  "#922B21",
    "Australia": "#D35400",
}


# ── Logarithmic model ──────────────────────────────────────────────────
def log_func(x, a, b):
    """y = a + b * log(x)"""
    return a + b * np.log(x)


# ── Load data ───────────────────────────────────────────────────────────
records = []
for country, city_folder, display_name, flag_code in CITIES:
    csv_path = DATA_ROOT / country / city_folder / "Fuse" / "Multi_Concat" / "results.csv"
    if not csv_path.exists():
        print(f"WARNING: missing {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    # Mean R2_true across all indicators and folds, per ratio
    agg = df.groupby("ratio")["R2_true"].mean().reset_index()
    agg.columns = ["ratio", "mean_r2"]
    agg["country"] = country
    agg["city"] = display_name
    agg["flag"] = flag_code
    records.append(agg)

data = pd.concat(records, ignore_index=True)

# ── Determine grid layout ──────────────────────────────────────────────
n_cities = len(CITIES)
ncols = 4
nrows = int(np.ceil(n_cities / ncols))

# ── Plot ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": FONT_NAME,
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(7.2, nrows * 1.55),
    sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.45, "wspace": 0.22},
)
axes = axes.flatten()

for idx, (country, city_folder, display_name, flag_code) in enumerate(CITIES):
    ax = axes[idx]
    sub = data[(data["city"] == display_name) & (data["country"] == country)].sort_values("ratio")
    color = COUNTRY_COLORS[country]

    x_raw = sub["ratio"].values
    y_raw = sub["mean_r2"].values

    # Scatter: raw data points (small, semi-transparent)
    ax.scatter(
        x_raw * 100, y_raw,
        s=12, color=color, alpha=0.4, edgecolors="none", zorder=3,
    )

    # Logarithmic curve fit: y = a + b * log(x)
    if len(x_raw) >= 2:
        try:
            popt, _ = curve_fit(log_func, x_raw, y_raw, p0=[0.5, 0.1], maxfev=5000)
            x_smooth = np.linspace(x_raw.min(), x_raw.max(), 200)
            y_smooth = log_func(x_smooth, *popt)
            ax.plot(
                x_smooth * 100, y_smooth,
                linewidth=1.4, color=color, zorder=4,
            )
        except RuntimeError:
            # Fallback: just connect the points if curve_fit fails
            ax.plot(x_raw * 100, y_raw, linewidth=1.0, color=color, zorder=4)

    # Horizontal reference line at R² = 0.8
    ax.axhline(0.8, color="#888888", linestyle="--", linewidth=0.7, zorder=1)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 120, 20))
    ax.set_yticks(np.arange(0.0, 1.2, 0.2))

    # Flag + city name as title
    flag_path = FLAG_DIR / f"{flag_code}.png"
    if flag_path.exists():
        flag_img = Image.open(flag_path).convert("RGBA")
        flag_img = flag_img.resize((20, int(20 * flag_img.height / flag_img.width)), Image.LANCZOS)
        imagebox = OffsetImage(np.array(flag_img), zoom=0.65)
        ab = AnnotationBbox(
            imagebox,
            (0.08, 1.02),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0.5, 0.0),
        )
        ax.add_artist(ab)
        ax.set_title(f"  {display_name}", fontsize=8, fontweight="bold", pad=8)
    else:
        ax.set_title(display_name, fontsize=8, fontweight="bold", pad=4)

    # Only show axis labels on edges
    if idx >= (nrows - 1) * ncols or idx >= n_cities - ncols:
        ax.set_xlabel("Sampling ratio (%)", fontsize=8)
    if idx % ncols == 0:
        ax.set_ylabel("Mean R\u00b2", fontsize=8)

    ax.tick_params(direction="in")

# Hide unused panels
for idx in range(n_cities, len(axes)):
    axes[idx].set_visible(False)

# ── Save ────────────────────────────────────────────────────────────────
out_png = OUT_DIR / "scaling_curves_cities.png"
out_pdf = OUT_DIR / "scaling_curves_cities.pdf"
fig.savefig(str(out_png), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(str(out_pdf), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
