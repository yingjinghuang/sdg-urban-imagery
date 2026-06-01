"""Within-modal fusion: self ⊕ spatial inside one modality.

For each modality (SV, RS) produces:

    {modality}/Concat_self_spatial.pkl
        Mocov3VITB-self ⊕ Mocov3VITB-spatial                 1536-d

This is the input used by ``scripts/run_fold_single_modal.sh`` to produce
the Fig 2a single-modal bars (SV-only and RS-only).

Replaces the original ``concat_single_modal.py``, which hardcoded
``modal = 'RS'`` (or ``'SV'``) at the bottom of the script and required
re-editing between runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.fuse._concat import hstack_on_geoid
from src.fuse.concat_cross_modal import mocov3_filename


# Convention: SV is pretrained for 100 epochs (ep99), RS for 50 (ep49).
DEFAULT_EPOCH = {"SV": 99, "RS": 49}


def fuse_within(root: str, country: str, city: str, sv_epoch: int, rs_epoch: int) -> None:
    base = Path(root) / country / city
    epochs = {"SV": sv_epoch, "RS": rs_epoch}

    for modality in ("SV", "RS"):
        ep = epochs[modality]
        self_file = str(base / modality / mocov3_filename(modality, "self", country, city, ep))
        spat_file = str(base / modality / mocov3_filename(modality, "spatial", country, city, ep))
        output = str(base / modality / "Concat_self_spatial.pkl")
        print(f"[fuse-within] {country}/{city}/{modality}")
        hstack_on_geoid([self_file, spat_file], output, expected_dim_per_block=768)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="Unit features root (FEATURES_UNIT_DIR).")
    p.add_argument("--country", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--sv_epoch", type=int, default=DEFAULT_EPOCH["SV"])
    p.add_argument("--rs_epoch", type=int, default=DEFAULT_EPOCH["RS"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fuse_within(args.root, args.country, args.city, args.sv_epoch, args.rs_epoch)


if __name__ == "__main__":
    main()
