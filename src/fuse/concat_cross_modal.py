"""Cross-modal fusion: SV ⊕ RS at the neighborhood level.

Produces four Fuse-level feature files for each (Country, City):

    Fuse/Concat_self.pkl
        self-SV ⊕ self-RS                                    1536-d

    Fuse/Concat_spatial.pkl
        spatial-SV ⊕ spatial-RS                              1536-d

    Fuse/Concat_spatial_self.pkl       [MAIN FRAMEWORK]
        spatial-SV ⊕ spatial-RS ⊕ self-SV ⊕ self-RS          3072-d

    Fuse/Concat_ImageNet.pkl           [Fig 1a baseline]
        ImageNet-SV ⊕ ImageNet-RS                            1536-d

Replaces the original ``concat_sv_rs.py`` + ``concat_self_spatial.py``,
which inlined the city list and hardcoded variant choice via commented
lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.fuse._concat import hstack_on_geoid


def mocov3_filename(modality: str, variant: str, country: str, city: str, epoch: int) -> str:
    return f"Mocov3VITB-{variant}-{country}-{city}-ep{epoch}.pkl"


def fuse_all(root: str, country: str, city: str, sv_epoch: int, rs_epoch: int) -> None:
    base = Path(root) / country / city
    fuse_dir = base / "Fuse"

    def sv(name: str) -> str: return str(base / "SV" / name)
    def rs(name: str) -> str: return str(base / "RS" / name)

    def mocov3(modality: str, variant: str, epoch: int) -> str:
        return str(base / modality / mocov3_filename(modality, variant, country, city, epoch))

    print(f"[fuse-cross] {country}/{city}")

    # Concat_self
    hstack_on_geoid(
        [mocov3("SV", "self", sv_epoch), mocov3("RS", "self", rs_epoch)],
        str(fuse_dir / "Concat_self.pkl"),
        expected_dim_per_block=768,
    )

    # Concat_spatial
    hstack_on_geoid(
        [mocov3("SV", "spatial", sv_epoch), mocov3("RS", "spatial", rs_epoch)],
        str(fuse_dir / "Concat_spatial.pkl"),
        expected_dim_per_block=768,
    )

    # Concat_spatial_self (main framework — 4-way concat)
    hstack_on_geoid(
        [
            mocov3("SV", "spatial", sv_epoch),
            mocov3("RS", "spatial", rs_epoch),
            mocov3("SV", "self", sv_epoch),
            mocov3("RS", "self", rs_epoch),
        ],
        str(fuse_dir / "Concat_spatial_self.pkl"),
        expected_dim_per_block=768,
    )

    # Concat_ImageNet
    hstack_on_geoid(
        [sv("ImageNet.pkl"), rs("ImageNet.pkl")],
        str(fuse_dir / "Concat_ImageNet.pkl"),
        expected_dim_per_block=768,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="Unit features root (FEATURES_UNIT_DIR).")
    p.add_argument("--country", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--sv_epoch", type=int, default=99)
    p.add_argument("--rs_epoch", type=int, default=49)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fuse_all(args.root, args.country, args.city, args.sv_epoch, args.rs_epoch)


if __name__ == "__main__":
    main()
