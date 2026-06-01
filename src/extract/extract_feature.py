"""Per-image feature extraction with a pretrained ViT-B backbone.

Loads a Mocov3 / ImageNet ViT-B checkpoint, runs every image listed in a
manifest pickle (``[{"path": "..."}, ...]``) through the encoder, and
writes the 768-dim features to an HDF5 file in chunks.

Multi-GPU via ``torch.nn.DataParallel`` if available. Mixed precision via
``autocast``. Restarts from completed chunks if the output already exists.

Usage:
    python -m src.extract.extract_feature \
        --pretrained_model_path ${MOCOV3_CKPT_DIR}/Mocov3VITB-spatial-cbg-Adelaide/checkpoint_99.pth.tar \
        --data_path ${PROCESSED_DIR}/Australia/Adelaide/paths.pkl \
        --save_path ${FEATURES_RAW_DIR}/Australia/Adelaide/SV/Mocov3VITB-spatial-Australia-Adelaide-ep99.h5

If ``--pretrained_model_path imagenet`` is given, the script loads a
torchvision ImageNet-pretrained ViT-B (no contrastive finetuning). This
provides the Fig 1a ImageNet baseline features.
"""

from __future__ import annotations

import argparse
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.cuda.amp import autocast
from tqdm import tqdm

# Local: the same ViT definitions used during pretraining.
# See pretrain/vits.py — extraction and pretraining must use identical models.
from pretrain.vits import vit_base

from src._common.paths import remap_image_paths

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- Data loading -----------------------------------------------------------


class ImageDataset(data.Dataset):
    """Loads RGB images from a path list, returning (tensor, path, index)."""

    def __init__(self, paths: list[str], indices: list[str], transform):
        self.paths = paths
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.paths[i], self.indices[i]


def load_image_list(manifest_path: str) -> list[str]:
    """Read a manifest pickle and return the list of image paths."""
    df = pd.read_pickle(manifest_path)
    paths = remap_image_paths(df["path"]).tolist()
    print(f"[extract] manifest: {manifest_path} -> {len(paths)} images")
    return paths


# --- Model loading ----------------------------------------------------------


def load_mocov3_vit_b(checkpoint_path: str) -> nn.Module:
    """Load a Mocov3 ViT-B checkpoint into a vit_base model."""
    print(f"[extract] loading Mocov3 ViT-B from {checkpoint_path}")
    model = vit_base()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    # Keep only the base_encoder weights, strip prefix.
    linear_keyword = "head"
    prefix = "module.base_encoder."
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith(prefix) and not k.startswith(prefix + linear_keyword):
            cleaned[k[len(prefix):]] = v
    msg = model.load_state_dict(cleaned, strict=False)
    expected_missing = {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}
    assert set(msg.missing_keys) == expected_missing, (
        f"Unexpected missing keys: {set(msg.missing_keys) - expected_missing}"
    )
    return model


def load_imagenet_vit_b() -> nn.Module:
    """Load a torchvision ImageNet-pretrained ViT-B (baseline for Fig 1a)."""
    import torchvision.models as tvm

    print("[extract] loading ImageNet ViT-B/16")
    model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
    # Strip the classification head to return penultimate features (768-d).
    model.heads = nn.Identity()
    return model


def build_model(pretrained_model_path: str) -> nn.Module:
    if pretrained_model_path.lower() == "imagenet":
        return load_imagenet_vit_b()
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_model_path}")
    return load_mocov3_vit_b(pretrained_model_path)


# --- Extraction loop --------------------------------------------------------


def extract_features(
    model: nn.Module,
    image_paths: list[str],
    save_path: str,
    *,
    batch_size: int = 1024,
    num_workers: int = 16,
    save_every: int = 50_000,
    feature_dim: int = 768,
) -> None:
    """Stream features to HDF5 in chunks of ``save_every`` rows."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda" if n_gpus > 0 else "cpu")
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_paths, image_paths, transform)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )

    buffer = pd.DataFrame()
    part = 0
    with torch.no_grad():
        for images, paths, indices in tqdm(loader, desc="extract"):
            keep = [i for i, (p, idx) in enumerate(zip(paths, indices)) if p and idx]
            if not keep:
                continue
            images = images[keep].to(device, non_blocking=True)
            with autocast():
                feats = model(images).float().cpu().numpy()
            chunk = pd.DataFrame(feats.reshape(len(keep), feature_dim))
            chunk["index"] = [indices[i] for i in keep]
            buffer = pd.concat([buffer, chunk], ignore_index=True) if not buffer.empty else chunk
            if buffer.shape[0] >= save_every:
                buffer, part = _flush(buffer, save_path, part)
    _flush(buffer, save_path, part)


def _flush(buffer: pd.DataFrame, save_path: str, part: int) -> tuple[pd.DataFrame, int]:
    if buffer.empty:
        return buffer, part
    buffer.to_hdf(save_path, key=f"part_{part}", mode="a")
    print(f"[extract] wrote part_{part} ({len(buffer)} rows) -> {save_path}")
    return pd.DataFrame(), part + 1


# --- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pretrained_model_path", required=True,
                   help='Path to a Mocov3 checkpoint, or the literal "imagenet".')
    p.add_argument("--data_path", required=True, help="Image-manifest pickle.")
    p.add_argument("--save_path", required=True, help="Output HDF5 path.")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--save_every", type=int, default=50_000)
    p.add_argument("--num_workers", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_image_list(args.data_path)
    if not paths:
        print(f"[extract] no images for {args.data_path}; nothing to do")
        return
    model = build_model(args.pretrained_model_path)
    extract_features(
        model,
        paths,
        args.save_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
