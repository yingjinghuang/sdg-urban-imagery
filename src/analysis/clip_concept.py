"""CLIP zero-shot concept scoring for every image in a manifest.

Loads OpenAI CLIP-ViT-B/32, scores each image against a set of named
visual concepts using prompts of the form ``"A photo of a {concept} street"``,
and writes a softmax-normalized score table.

Output: a pickle with one row per image — columns are ``path`` plus one
column per concept name. This file is consumed by
``figures/_data_prep/prep_fig2bc_clip_concept.py`` to produce the
Figure 2b–c bars.

Replaces ``3_5analysis/clip_concept.py``. Cleanup: hardcoded paths
replaced with CLI args; three rotated concept-set drafts (commented in
the original) consolidated into one configurable ``--concepts`` flag;
default model identifier now the HuggingFace hub id instead of a local
cache path.

Usage:
    python -m src.analysis.clip_concept \
        --input_pkl  ${PROCESSED_DIR}/sv_paths_all.pkl \
        --output_pkl ${PROCESSED_DIR}/sv_clip_scores.pkl \
        --concepts   "intersection,roundabout,parking lot,court,factory,bridge,rooftop,grid,maze,organic,uniform,residential,industrial,slum"
"""

from __future__ import annotations

import argparse
import os

# CLIP processors can fork lots of tokenizer workers if left to their own
# devices, which slows large multi-GPU runs. Disable that here.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# Default concept set used in the published Figure 2b–c.
# (Two paired groups: noun-like objects for the self-contrastive bars,
# scene-attribute concepts for the spatial-contrastive bars.)
DEFAULT_CONCEPTS = (
    "intersection,roundabout,parking lot,court,factory,bridge,rooftop,"
    "grid,maze,organic,uniform,residential,industrial,slum"
)


# --- Model wrapper ---------------------------------------------------------


class ImageEncoder(nn.Module):
    """Expose only the image branch so ``DataParallel`` can shard cleanly."""

    def __init__(self, clip_model: CLIPModel):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.clip_model.get_image_features(pixel_values=pixel_values)
        return feats / feats.norm(p=2, dim=-1, keepdim=True)


# --- Dataset ---------------------------------------------------------------


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Image.Image:
        try:
            return Image.open(self.paths[i]).convert("RGB")
        except Exception as e:
            print(f"[clip] failed to load {self.paths[i]}: {e}")
            return Image.new("RGB", (224, 224), color="black")


class _ProcessorCollate:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, batch: list[Image.Image]) -> torch.Tensor:
        out = self.processor(images=batch, return_tensors="pt")
        return out["pixel_values"]


# --- Scoring ---------------------------------------------------------------


def encode_text(model: nn.Module, processor: CLIPProcessor, concepts: list[str], device: str
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [f"A photo of a {c} street" for c in concepts]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    base = model.module.clip_model if isinstance(model, nn.DataParallel) else model.clip_model
    with torch.no_grad():
        text_feats = base.get_text_features(**inputs)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        logit_scale = base.logit_scale.exp()
    return text_feats, logit_scale


def score_images(model: nn.Module, loader: DataLoader,
                  text_feats: torch.Tensor, logit_scale: torch.Tensor, device: str
                  ) -> np.ndarray:
    out = []
    with torch.no_grad():
        for pixel_values in tqdm(loader, desc="clip"):
            pixel_values = pixel_values.to(device)
            image_feats = model(pixel_values)
            # (B, T) softmaxed; ``probs[i, j]`` is the relative match of image i
            # to concept j among the supplied concepts.
            logits = logit_scale * image_feats @ text_feats.t()
            out.append(logits.softmax(dim=-1).cpu().numpy())
    return np.vstack(out)


# --- CLI -------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] device={device}")

    print(f"[clip] loading model: {args.model_path}")
    base = CLIPModel.from_pretrained(args.model_path)
    processor = CLIPProcessor.from_pretrained(args.model_path)
    model = ImageEncoder(base).to(device)
    if torch.cuda.device_count() > 1:
        print(f"[clip] using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.eval()

    print(f"[clip] reading {args.input_pkl}")
    df = pd.read_pickle(args.input_pkl)
    paths = [p for p in df["path"].tolist() if os.path.exists(p)]
    print(f"[clip] {len(paths)} images exist on disk")

    dataset = ImagePathDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_ProcessorCollate(processor),
        pin_memory=True,
    )

    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    text_feats, logit_scale = encode_text(model, processor, concepts, device)
    scores = score_images(model, loader, text_feats, logit_scale, device)

    score_df = pd.DataFrame(scores, columns=concepts)
    out = pd.concat([pd.Series(paths, name="path"), score_df], axis=1)
    os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
    out.to_pickle(args.output_pkl)
    print(f"[clip] saved {args.output_pkl}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input_pkl", required=True,
                   help="Pickle with a 'path' column listing image paths.")
    p.add_argument("--output_pkl", required=True,
                   help="Output pickle (path + one column per concept).")
    p.add_argument("--model_path", default="openai/clip-vit-base-patch32",
                   help="HuggingFace CLIP model identifier or local cache dir.")
    p.add_argument("--concepts", default=DEFAULT_CONCEPTS,
                   help="Comma-separated concept names (used in 'A photo of a {} street').")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main()
