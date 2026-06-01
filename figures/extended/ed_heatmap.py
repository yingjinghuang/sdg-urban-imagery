import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import os
from pathlib import Path
from matplotlib import font_manager

# Font setup - Arial
arial_candidates = [
    'C:/Windows/Fonts/arial.ttf',
    'C:/Windows/Fonts/Arial.ttf',
]
for fp in arial_candidates:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = 'Arial'
        break
else:
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'

# Flag paths
flag_dir = Path("D:/Workspace/01Project/00_SDG/images/src/flags")
flag_paths = {
    "Australia": flag_dir / "au.png",
    "Brazil": flag_dir / "br.png",
    "China": flag_dir / "cn.png",
    "France": flag_dir / "fr.png",
    "Nigeria": flag_dir / "ng.png",
    "Portugal": flag_dir / "pt.png",
    "US": flag_dir / "us.png",
}

# SDG mapping for each country's indicators
sdg_mapping = {
    # Australia
    'med_hhinc': 'SDG 1', 'age65': 'SDG 3', 'arthritis': 'SDG 3', 'asthma': 'SDG 3',
    'cancer': 'SDG 3', 'diabetes': 'SDG 3', 'heart_disease': 'SDG 3', 'kidney_disease': 'SDG 3',
    'lung_condition': 'SDG 3', 'mental_health': 'SDG 3',
    'edu_year12': 'SDG 4', 'edu_noschool': 'SDG 4',
    'mean_capgain': 'SDG 8', 'med_capgain': 'SDG 8', 'unemploy': 'SDG 8',
    'internet': 'SDG 9',
    'popden': 'SDG 11', 'renter': 'SDG 11',
    'drive': 'SDG 13', 'publictrans': 'SDG 13', 'walk': 'SDG 13', 'bike': 'SDG 13',
    # Brazil
    'BR01': 'SDG 1', 'BR02': 'SDG 3', 'BR12': 'SDG 4', 'BR13': 'SDG 5',
    'BR14': 'SDG 10', 'BR17': 'SDG 11',
    # Hong Kong
    'HK02': 'SDG 1', 'HK06': 'SDG 4', 'HK07': 'SDG 8', 'HK15': 'SDG 10', 'HK17': 'SDG 16',
    # France
    'FR02': 'SDG 3', 'FR03': 'SDG 8', 'FR04': 'SDG 8',
    'FR08': 'SDG 10', 'FR09': 'SDG 11', 'FR11': 'SDG 11',
    # Nigeria
    'pub_health': 'SDG 3', 'health_den': 'SDG 6', 'market_den': 'SDG 8',
    # Portugal
    'PT16': 'SDG 4', 'PT24': 'SDG 8', 'PT28': 'SDG 8',
    'PT33': 'SDG 10', 'PT34': 'SDG 11', 'PT35': 'SDG 11',
    'PT37': 'SDG 16', 'PT38': 'SDG 16',
    # US
    'logincome': 'SDG 1', 'povertyline_below100': 'SDG 1', 'povertyline_below200': 'SDG 1',
    'cancercrud': 'SDG 3', 'diabetescr': 'SDG 3', 'obesitycru': 'SDG 3',
    'lpacrudepr': 'SDG 3', 'mhlthcrude': 'SDG 3', 'phlthcrude': 'SDG 3',
    'drove_alone_per_cbg': 'SDG 13', 'publictrans_per_cbg': 'SDG 13',
    'walkbike_per_cbg': 'SDG 13', 'estpmiles': 'SDG 13', 'estptrp': 'SDG 13',
    'estvmiles': 'SDG 13', 'estvtrp': 'SDG 13',
    'logcrime': 'SDG 16', 'logpetty': 'SDG 16',
}

base_path = Path("../../data/regression_outputs/regmodels/Fold")

# Collect all results
all_results = []
for country_dir in sorted(base_path.iterdir()):
    if not country_dir.is_dir():
        continue
    country = country_dir.name
    for city_dir in sorted(country_dir.iterdir()):
        if not city_dir.is_dir() or city_dir.name == 'baseline':
            continue
        city = city_dir.name
        display_city = f"{country} (All)" if city == "All" else city
        results_file = city_dir / "Fuse" / "Multi_Concat" / "results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            for _, row in df.iterrows():
                target = row['target']
                sdg = sdg_mapping.get(target, 'Unknown')
                all_results.append({
                    'country': country,
                    'city': display_city,
                    'target': target,
                    'sdg': sdg,
                    'r2': row['r2']
                })

df_all = pd.DataFrame(all_results)

# Compute mean R2 per city per SDG
pivot = df_all.groupby(['city', 'sdg'])['r2'].mean().reset_index()
pivot_table = pivot.pivot(index='city', columns='sdg', values='r2')

# Order SDGs
sdg_order = ['SDG 1', 'SDG 3', 'SDG 4', 'SDG 5', 'SDG 6', 'SDG 8', 'SDG 9', 'SDG 10', 'SDG 11', 'SDG 13', 'SDG 16']
sdg_order = [s for s in sdg_order if s in pivot_table.columns]
pivot_table = pivot_table[sdg_order]

# Order cities by country then city name
city_country = df_all[['city', 'country']].drop_duplicates().sort_values(['country', 'city'])
city_order = city_country['city'].tolist()
pivot_table = pivot_table.reindex(city_order)

# Create city display names
city_display = []
country_list = []
for _, row in city_country.iterrows():
    if 'All' in row['city']:
        city_display.append(row['country'])
    else:
        city_display.append(row['city'])
    country_list.append(row['country'])

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
mask = pivot_table.isna()

# Fill missing cells with light gray background
ax.set_facecolor('#f0f0f0')

sns.heatmap(pivot_table,
            annot=True, fmt='.2f',
            annot_kws={'fontsize': 9},
            cmap='RdYlGn',
            mask=mask,
            vmin=0, vmax=0.9,
            linewidths=0.8, linecolor='white',
            xticklabels=sdg_order,
            yticklabels=city_display,
            cbar_kws={'label': '$R^2$', 'shrink': 0.8},
            ax=ax)

# Add hatching to missing cells
for i in range(len(pivot_table)):
    for j in range(len(sdg_order)):
        if mask.iloc[i, j]:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                        hatch='///', edgecolor='#cccccc', linewidth=0))

ax.set_xlabel('SDG Category', fontsize=12, fontweight='bold')
ax.set_ylabel('')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=10)

# Add flag icons next to y-axis labels
for i, (city_name, country_name) in enumerate(zip(city_display, country_list)):
    flag_file = flag_paths.get(country_name)
    if flag_file and flag_file.exists():
        flag_img = mpimg.imread(str(flag_file))
        imagebox = OffsetImage(flag_img, zoom=0.02)
        ab = AnnotationBbox(imagebox, (-0.15, i + 0.5),
                           xycoords=('axes fraction', 'data'),
                           frameon=False,
                           box_alignment=(1.0, 0.5))
        ax.add_artist(ab)

# Add country separators
prev_country = None
for i, c in enumerate(country_list):
    if prev_country is not None and c != prev_country:
        ax.axhline(y=i, color='gray', linewidth=2, linestyle='-')
    prev_country = c

plt.tight_layout()
plt.subplots_adjust(left=0.22)

output_path = "D:/Workspace/01Project/00_SDG/images/extended/heatmap_city_sdg.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved to {output_path}")
