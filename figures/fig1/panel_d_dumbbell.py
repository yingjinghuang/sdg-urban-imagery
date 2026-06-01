"""
Dumbbell plot comparing minimum sampling ratio to reach R²=0.8
under random vs strategic sampling for each city.
"""

import pathlib
import warnings
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from PIL import Image
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────
RANDOM_ROOT = pathlib.Path("../../data/regression_outputs/regmodels/Ratio")
STRATEGIC_ROOT = pathlib.Path("../../data/regression_outputs/regmodels/Sampling")
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

R2_TARGET = 0.8


# ── Helper functions ────────────────────────────────────────────────────
def log_func(x, a, b):
    """Logarithmic model: y = a * log(x) + b."""
    return a * np.log(x) + b


def find_min_ratio(csv_path: pathlib.Path, target_r2: float = R2_TARGET) -> float:
    """
    Return the minimum sampling ratio at which overall R² reaches *target_r2*.

    Strategy:
    1. Read the CSV; compute mean R2_true across all indicators (and folds
       if present) at each ratio.
    2. Fit  y = a*log(x) + b  to the (ratio, mean_R2_true) pairs.
    3. Solve for x where y = target_r2.  Return >0.9 sentinel (0.95) if the
       target is never reached within [0.1, 0.9].
    """
    df = pd.read_csv(csv_path)
    # Mean R2_true per ratio (across all indicators and folds)
    mean_r2 = df.groupby("ratio")["R2_true"].mean().sort_index()
    ratios = mean_r2.index.values.astype(float)
    values = mean_r2.values.astype(float)

    # If already reached at lowest ratio
    if values[0] >= target_r2:
        return ratios[0]

    # If never reached at highest ratio, return sentinel
    if values[-1] < target_r2:
        return 0.95  # sentinel for ">0.9"

    try:
        popt, _ = curve_fit(log_func, ratios, values, p0=[0.3, 0.7], maxfev=5000)
        a, b = popt
        # Solve a*log(x) + b = target_r2  =>  x = exp((target_r2 - b) / a)
        if a == 0:
            return 0.95
        x_sol = np.exp((target_r2 - b) / a)
        # Clamp to reasonable range
        if x_sol < 0.05:
            return 0.05
        if x_sol > 0.9:
            return 0.95
        return float(x_sol)
    except Exception:
        # Fallback: linear interpolation
        for i in range(len(values) - 1):
            if values[i] < target_r2 <= values[i + 1]:
                frac = (target_r2 - values[i]) / (values[i + 1] - values[i])
                return float(ratios[i] + frac * (ratios[i + 1] - ratios[i]))
        return 0.95


def load_flag(flag_code: str, target_height: int = 18) -> np.ndarray:
    """Load and resize a flag image."""
    img = Image.open(FLAG_DIR / f"{flag_code}.png").convert("RGBA")
    aspect = img.width / img.height
    new_w = int(target_height * aspect)
    img = img.resize((new_w, target_height), Image.LANCZOS)
    return np.array(img)


# ── Compute minimum ratios ─────────────────────────────────────────────
results = []
for country, folder, display, flag in CITIES:
    random_csv = RANDOM_ROOT / country / folder / "Fuse" / "Multi_Concat" / "results.csv"
    strategic_csv = STRATEGIC_ROOT / country / folder / "Fuse" / "Multi_Concat_pcahierachy" / "results.csv"

    r_random = find_min_ratio(random_csv) if random_csv.exists() else np.nan
    r_strategic = find_min_ratio(strategic_csv) if strategic_csv.exists() else np.nan

    results.append({
        "country": country,
        "folder": folder,
        "display": display,
        "flag": flag,
        "random": r_random,
        "strategic": r_strategic,
    })
    print(f"  {display:20s}  random={r_random:.3f}  strategic={r_strategic:.3f}")

df_res = pd.DataFrame(results)

# ── Plot ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": FONT_NAME,
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.0,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
})

n_cities = len(df_res)
fig_h = max(5, 0.38 * n_cities + 1.0)
fig, ax = plt.subplots(figsize=(5.5, fig_h))

# Colors
COL_RANDOM = "#E8792B"     # orange
COL_STRATEGIC = "#3C78D8"  # blue

y_positions = list(range(n_cities))[::-1]  # top-to-bottom

# Track country boundaries for separators
prev_country = None
separator_ys = []

for idx, (_, row) in enumerate(df_res.iterrows()):
    y = y_positions[idx]

    # Country separator
    if prev_country is not None and row["country"] != prev_country:
        sep_y = y + 0.5
        separator_ys.append(sep_y)
    prev_country = row["country"]

    r_rand = row["random"]
    r_strat = row["strategic"]

    # Connector line (only if both values are valid)
    if not (np.isnan(r_rand) or np.isnan(r_strat)):
        vals = [r_rand if r_rand < 0.95 else 0.93,
                r_strat if r_strat < 0.95 else 0.93]
        ax.plot([min(vals), max(vals)], [y, y], color="#AAAAAA", linewidth=1.2, zorder=1)

    # Dots
    marker_size = 50
    for val, color in [(r_rand, COL_RANDOM), (r_strat, COL_STRATEGIC)]:
        if np.isnan(val):
            continue
        if val >= 0.95:
            ax.scatter(0.93, y, s=marker_size, color=color, marker="D",
                       edgecolors=color, linewidths=1.0, zorder=3, alpha=0.6)
        else:
            ax.scatter(val, y, s=marker_size, color=color, edgecolors="white",
                       linewidths=0.5, zorder=3)

# Draw separators
for sep_y in separator_ys:
    ax.axhline(sep_y, color="#CCCCCC", linewidth=0.5, linestyle="--", zorder=0)

# Y-axis: city labels with flag icons
ax.set_yticks(y_positions)
ax.set_yticklabels([row["display"] for _, row in df_res.iterrows()], fontsize=8.5)

# Add flag icons to the left of city names
for idx, (_, row) in enumerate(df_res.iterrows()):
    y = y_positions[idx]
    flag_img = load_flag(row["flag"], target_height=12)
    im = OffsetImage(flag_img, zoom=1.0)
    # Place flag to the left of the y-axis label
    ab = AnnotationBbox(im, (0, y), xybox=(-68, 0), xycoords=("axes fraction", "data"),
                        boxcoords="offset points", frameon=False, pad=0)
    ax.add_artist(ab)

# X-axis
ax.set_xlim(0.05, 1.0)
ax.set_xticks(np.arange(0.1, 1.0, 0.1))
ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0.1, 1.0, 0.1)], fontsize=8)
ax.set_xlabel("Minimum sampling ratio to reach R² = 0.8", fontsize=9.5,
              fontfamily=FONT_NAME)

# Y-axis limits
ax.set_ylim(-0.8, n_cities - 0.2)

# Remove right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# Light grid on x-axis
ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_RANDOM,
           markersize=7, label="Random sampling"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_STRATEGIC,
           markersize=7, label="Strategic sampling"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888",
           markersize=5.5, alpha=0.6, label="> 0.9 (not reached)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=7.5,
          frameon=True, fancybox=False, edgecolor="#CCCCCC",
          borderpad=0.6, handletextpad=0.4)

plt.tight_layout()
plt.subplots_adjust(left=0.30)

# ── Save ────────────────────────────────────────────────────────────────
out_png = OUT_DIR / "dumbbell_sampling.png"
out_pdf = OUT_DIR / "dumbbell_sampling.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close(fig)
