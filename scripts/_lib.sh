# scripts/_lib.sh
# Common environment and helpers for all bash launchers in this repo.
#
# Source this from every script:
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     source "${SCRIPT_DIR}/_lib.sh"
#
# It exports:
#   REPO_ROOT, CONFIG_PATH, CITIES_PATH, PYTHON
#   DATA_ROOT, OUTPUT_ROOT, PROCESSED_DIR, FEATURES_RAW_DIR, FEATURES_UNIT_DIR,
#   LABELS_DIR, SEG_RESULTS_DIR, PRETRAIN_CKPT_DIR, REGRESSION_OUT_DIR,
#   CUDA_DEVICES, RAW_SV_DIR, RAW_RS_DIR
#
# It provides:
#   country_targets <COUNTRY>     -> echoes comma-separated target list
#   for_each_region <CMD>         -> runs CMD with COUNTRY,CITY env per region

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${SDG_CONFIG:-${REPO_ROOT}/configs/paths.yaml}"
CITIES_PATH="${REPO_ROOT}/configs/cities.yaml"
PYTHON="${PYTHON:-python}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[_lib.sh] paths.yaml not found at: ${CONFIG_PATH}" >&2
    echo "[_lib.sh] Copy configs/paths.example.yaml to configs/paths.yaml and edit." >&2
    exit 1
fi

# --- Load paths.yaml into shell variables -----------------------------------
# Single-shot inline Python to avoid needing yq. Exports each key as
# PATHS_<KEY_UPPER>. Resolves ${data_root} and ${output_root} interpolations.

eval "$(${PYTHON} - "${CONFIG_PATH}" <<'PYEOF'
import sys, yaml, re
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
# Interpolate ${var}
def resolve(value, scope):
    if not isinstance(value, str):
        return value
    pattern = re.compile(r'\$\{([^}]+)\}')
    while True:
        m = pattern.search(value)
        if not m:
            return value
        key = m.group(1)
        if key not in scope:
            raise KeyError(f"Unresolved variable ${{{key}}} in paths.yaml")
        value = value.replace(m.group(0), str(scope[key]))
resolved = {}
for k, v in cfg.items():
    resolved[k] = resolve(v, resolved)
for k, v in resolved.items():
    key = k.upper()
    # Quote for shell safety
    sv = str(v).replace("'", "'\\''")
    print(f"export {key}='{sv}'")
PYEOF
)"

# Convenience aliases matching variable names used throughout the codebase.
export DATA_ROOT="${DATA_ROOT}"
export OUTPUT_ROOT="${OUTPUT_ROOT}"
export PROCESSED_DIR="${PROCESSED_DIR}"
export FEATURES_RAW_DIR="${FEATURES_RAW_DIR}"
export FEATURES_UNIT_DIR="${FEATURES_UNIT_DIR}"
export LABELS_DIR="${LABELS_DIR}"
export SEG_RESULTS_DIR="${SEG_RESULTS_DIR:-${DATA_ROOT}/seg_results}"
export MOCOV3_CKPT_DIR="${MOCOV3_CKPT_DIR}"
export PRETRAIN_OUT_DIR="${PRETRAIN_OUT_DIR}"
export REGRESSION_OUT_DIR="${REGRESSION_OUT_DIR}"
export RAW_SV_DIR="${RAW_SV_DIR}"
export RAW_RS_DIR="${RAW_RS_DIR}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"

# --- Helpers -----------------------------------------------------------------

# country_targets <Country> -> echo comma-separated indicator list from cities.yaml
country_targets() {
    local country="$1"
    ${PYTHON} - "${CITIES_PATH}" "${country}" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
country = sys.argv[2]
targets = cfg.get("targets", {}).get(country, [])
print(",".join(targets))
PYEOF
}

# for_each_region <command...>  -> calls command with COUNTRY and CITY set
# Example:  for_each_region run_my_step
for_each_region() {
    local cmd="$1"
    ${PYTHON} - "${CITIES_PATH}" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for country, regions in cfg["regions"].items():
    for r in regions:
        print(f"{country}\t{r['name']}")
PYEOF
    | while IFS=$'\t' read -r COUNTRY CITY; do
        export COUNTRY CITY
        TARGETS="$(country_targets "${COUNTRY}")"
        export TARGETS
        echo "[for_each_region] ${COUNTRY}/${CITY}"
        "${cmd}" "${COUNTRY}" "${CITY}"
    done
}

# Echo iteration without invoking a command (handy for ad-hoc loops).
list_regions() {
    ${PYTHON} - "${CITIES_PATH}" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for country, regions in cfg["regions"].items():
    for r in regions:
        print(f"{country} {r['name']} {r.get('sv_epoch', 99)} {r.get('rs_epoch', 49)}")
PYEOF
}
