"""CLIP zero-shot concept scoring for the manuscript concept probe.

The final manuscript probes Los Angeles imagery with modality-specific concept
sets (main Fig. 3b-c; Supplementary Fig. 59):

* satellite (RS): concrete, rooftop, vegetation, soil;
* street view (SV): seven concrete nouns (building, car, fence, pole, window,
  road, tree) and seven scene-character adjectives (chaotic, orderly,
  depressing, lively, safe, dilapidated, wealthy).

The script scores every image listed in the supplied manifest and writes a
pickle containing ``path`` plus one column per concept. The plotting workflow
uses the four concepts shown in the final main figure (RS: concrete, rooftop,
vegetation, soil; SV: building, window, road, tree).

The Supplementary Information records the street-view prompt template exactly
as ``A photo of a {concept} street`` and describes the satellite prompt as an
analogous satellite template. Because the exact historical satellite wording is
not stated in the manuscript, this release uses the transparent default
``A satellite photo of {concept}``; it can be overridden with
``--prompt-template`` when reconstructing a historical run.
"""

from __future__ import annotations

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


SV_CONCEPTS = [
    "building",
    "car",
    "fence",
    "pole",
    "window",
    "road",
    "tree",
    "chaotic",
    "orderly",
    "depressing",
    "lively",
    "safe",
    "dilapidated",
    "wealthy",
]
RS_CONCEPTS = ["concrete", "rooftop", "vegetation", "soil"]

DEFAULT_PROMPT_TEMPLATES = {
    "SV": "A photo of a {concept} street",
    "RS": "A satellite photo of {concept}",
}


class ImageEncoder(nn.Module):
    """Expose only the CLIP image branch so DataParallel can shard it."""

    def __init__(self, clip_model: CLIPModel):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.clip_model.get_image_features(pixel_values=pixel_values)
        return feats / feats.norm(p=2, dim=-1, keepdim=True)


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Image.Image:
        path = self.paths[i]
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load image: {path}") from exc


class _ProcessorCollate:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, batch: list[Image.Image]) -> torch.Tensor:
        return self.processor(images=batch, return_tensors="pt")["pixel_values"]


def manuscript_concepts(modality: str) -> list[str]:
    modality = modality.upper()
    if modality == "SV":
        return list(SV_CONCEPTS)
    if modality == "RS":
        return list(RS_CONCEPTS)
    raise ValueError(f"unsupported modality: {modality}")


def encode_text(
    model: nn.Module,
    processor: CLIPProcessor,
    concepts: list[str],
    prompt_template: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "{concept}" not in prompt_template:
        raise ValueError("--prompt-template must contain the literal placeholder {concept}")
    prompts = [prompt_template.format(concept=concept) for concept in concepts]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    base = model.module.clip_model if isinstance(model, nn.DataParallel) else model.clip_model
    with torch.no_grad():
        text_feats = base.get_text_features(**inputs)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        logit_scale = base.logit_scale.exp()
    return text_feats, logit_scale


def score_images(
    model: nn.Module,
    loader: DataLoader,
    text_feats: torch.Tensor,
    logit_scale: torch.Tensor,
    device: str,
) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for pixel_values in tqdm(loader, desc="clip"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            image_feats = model(pixel_values)
            logits = logit_scale * image_feats @ text_feats.t()
            rows.append(logits.softmax(dim=-1).cpu().numpy())
    if not rows:
        raise RuntimeError("no images were scored")
    return np.vstack(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-pkl", "--input_pkl", dest="input_pkl", required=True)
    parser.add_argument("--output-pkl", "--output_pkl", dest="output_pkl", required=True)
    parser.add_argument(
        "--modality",
        choices=["SV", "RS", "sv", "rs"],
        required=True,
        help="Imagery modality; selects the manuscript concept set and default prompt.",
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model identifier or local cache directory.",
    )
    parser.add_argument(
        "--concepts",
        default=None,
        help="Optional comma-separated override. By default uses the manuscript set for the modality.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Prompt containing {concept}. Defaults to the modality-specific release template.",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modality = args.modality.upper()
    concepts = (
        [value.strip() for value in args.concepts.split(",") if value.strip()]
        if args.concepts
        else manuscript_concepts(modality)
    )
    prompt_template = args.prompt_template or DEFAULT_PROMPT_TEMPLATES[modality]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] modality={modality} device={device}")
    print(f"[clip] concepts={concepts}")
    print(f"[clip] prompt_template={prompt_template!r}")
    print(f"[clip] loading model: {args.model_path}")

    base = CLIPModel.from_pretrained(args.model_path)
    processor = CLIPProcessor.from_pretrained(args.model_path)
    model = ImageEncoder(base).to(device)
    if torch.cuda.device_count() > 1:
        print(f"[clip] using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.eval()

    manifest = pd.read_pickle(args.input_pkl)
    if "path" not in manifest.columns:
        raise KeyError(f"{args.input_pkl} must contain a 'path' column")

    missing = [path for path in manifest["path"].tolist() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest images do not exist locally; first missing path: {missing[0]}"
        )
    paths = manifest["path"].tolist()
    print(f"[clip] scoring {len(paths)} images from {args.input_pkl}")

    loader = DataLoader(
        ImagePathDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_ProcessorCollate(processor),
        pin_memory=True,
    )

    text_feats, logit_scale = encode_text(
        model, processor, concepts, prompt_template, device
    )
    scores = score_images(model, loader, text_feats, logit_scale, device)

    output = pd.concat(
        [pd.Series(paths, name="path"), pd.DataFrame(scores, columns=concepts)], axis=1
    )
    os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
    output.to_pickle(args.output_pkl)
    print(f"[clip] saved {args.output_pkl} ({len(output)} rows)")


if __name__ == "__main__":
    main()
