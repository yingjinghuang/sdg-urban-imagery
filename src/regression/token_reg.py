"""Token-regression entry point.

Trains a Spatial Transformer regressor over per-modality visual tokens
(plus an optional geo token), iterating over every train/test split
defined in the community pickle (`set*` columns).

The command-line defaults match the regression configuration reported in
Supplementary Table 9 of the manuscript (hidden dimension 256, 8 attention
heads, 1 Transformer layer, batch size 128, 100 maximum epochs, 10-epoch
warmup, early-stopping patience 5, learning rate 1e-4, and weight decay
1e-4). Paper-reproduction launchers also pass these values explicitly so
that future changes to generic defaults cannot silently change the reported
configuration.

Outputs:
    {output_dir}/results.csv         per-split, per-target R²/MAE/MSE
    {output_dir}/results.h5          per-neighborhood predictions
    {output_dir}/S{tag}.pth.tar      best checkpoint per split
    {output_dir}/scaler_*.pkl        fitted feature/geo scalers

The community pickle determines the experiment shape:
    - 5-fold CV   -> set columns named setF0..setF4
    - Ratio sweep -> set01..set09  (used as ratios 0.1..0.9)
    - Sampling    -> setNN columns whose ratio is decoded from the name

This single script replaces three near-duplicate variants in the
original codebase (4regression_fold, 5regression_ratio,
6regression_sampling) and the `_non_spatial` mirrors — those collapse
into the ``--no_geo`` flag here.

Usage (called from bash/sdg/run_*.sh):
    python -m src.regression.token_reg \
        --feature_path .../Concat_spatial_self.pkl \
        --label_path   .../labels_norm.pkl \
        --community_path .../samples/ratio.pkl \
        --geo_path     .../geo.pkl \
        --targets      "logcrime,logpetty,..." \
        --output_dir   ${REGRESSION_OUT_DIR}/ratio/main/.../Token_Concat_spatial_self
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.regression._data import load_aligned
from src.regression._models import MaskedMSELoss, SpatialTransformerRegressor
from src.regression._train_utils import (
    EarlyStopping,
    evaluate,
    per_target_metrics,
    train_one_epoch,
)


def _split_tag(set_col: str) -> str:
    """Bare suffix of a ``set*`` column, used as the per-split file tag."""
    return set_col.replace("set", "")


def _maybe_ratio(set_col: str) -> float | None:
    """Decode a numeric ratio from a setNN column name, if present."""
    suffix = _split_tag(set_col)
    if suffix.isdigit():
        # e.g. "09" -> 0.9
        return int(suffix) / 10.0
    return None


def run(
    df_labels: pd.DataFrame,
    X_visual: np.ndarray,
    X_coords: np.ndarray | None,
    output_dir: str,
    targets: list[str],
    parameter: dict,
) -> None:
    use_geo = parameter["use_geo"]

    X = torch.tensor(X_visual, dtype=torch.float32)
    C = (
        torch.tensor(X_coords, dtype=torch.float32)
        if use_geo
        else torch.zeros((len(X), 2), dtype=torch.float32)
    )
    y = torch.tensor(df_labels[targets].values, dtype=torch.float32)
    dataset = TensorDataset(X, C, y)

    set_cols = sorted(c for c in df_labels.columns if str(c).startswith("set"))
    if not set_cols:
        raise ValueError("Community table has no 'set*' columns — nothing to iterate.")

    results: list[pd.DataFrame] = []
    for set_col in set_cols:
        tag = _split_tag(set_col)
        train_idx = df_labels[df_labels[set_col] == "train"].index
        test_idx = df_labels[df_labels[set_col] == "test"].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"[token_reg] skip {set_col}: empty train or test split")
            continue

        train_loader = DataLoader(Subset(dataset, train_idx),
                                   batch_size=parameter["batch_size"], shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx),
                                  batch_size=parameter["batch_size"], shuffle=False)

        model = SpatialTransformerRegressor(
            input_dim=parameter["feature_len"],
            num_heads=parameter["num_heads"],
            num_layers=parameter["num_layers"],
            hidden_dim=parameter["hidden_dim"],
            output_dim=y.size(1),
            use_geo=use_geo,
        ).to(parameter["device"])
        criterion = MaskedMSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=parameter["lr"],
                                     weight_decay=parameter["weight_decay"])
        early = EarlyStopping(patience=parameter["patience"], delta=0.005)
        ckpt_path = f"{output_dir}/S{tag}.pth.tar"

        for epoch in range(parameter["num_epochs"]):
            train_one_epoch(model, train_loader, criterion, optimizer, epoch, parameter)
            _, _, val_loss = evaluate(model, test_loader, criterion, parameter)
            early(val_loss, model, ckpt_path)
            if early.early_stop:
                print(f"[token_reg] {set_col}: early stop at epoch {epoch}")
                break

        # Final evaluation on the best checkpoint.
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        y_true, y_pred, _ = evaluate(model, test_loader, criterion, parameter)
        y_train = y[train_idx].cpu().numpy()
        epoch_metrics = per_target_metrics(y_true, y_pred, y_train, targets)
        epoch_metrics["set"] = set_col
        ratio = _maybe_ratio(set_col)
        if ratio is not None:
            epoch_metrics["ratio"] = ratio
        results.append(epoch_metrics)

        # Per-neighborhood predictions for downstream residual analysis.
        rec = df_labels[targets].copy()
        rec["set"] = "train"
        rec.loc[test_idx, "set"] = "test"
        for j, t in enumerate(targets):
            rec[f"pred_{t}"] = rec[t]
            rec.loc[test_idx, f"pred_{t}"] = y_pred[:, j]
        rec.to_hdf(f"{output_dir}/results.h5", key=f"S{tag}", mode="a")

        print(f"[token_reg] {set_col}: train={len(train_idx)} test={len(test_idx)} done")

    if results:
        out = pd.concat(results, ignore_index=True)
        out.to_csv(f"{output_dir}/results.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # I/O
    p.add_argument("--feature_path", required=True)
    p.add_argument("--label_path", required=True)
    p.add_argument("--community_path", required=True,
                   help="Pickle with GEOID + set* columns defining splits.")
    p.add_argument("--geo_path", default=None,
                   help="Pickle with GEOID + geometry (required unless --no_geo).")
    p.add_argument("--targets", required=True, help="Comma-separated target names.")
    p.add_argument("--output_dir", required=True)
    # Model — paper settings from Supplementary Table 9.
    p.add_argument("--feature_len", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=1)
    # Training — paper settings from Supplementary Table 9.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    # Device
    p.add_argument("--device", type=int, default=0)
    # Geo toggle (replaces token_reg_non_spatial.py)
    p.add_argument("--no_geo", action="store_true",
                   help="Disable the spatial coordinate branch.")
    # Optional run name
    p.add_argument("--mode", default="Fuse",
                   help="Bookkeeping label (informational; passed to scripts).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_geo = not args.no_geo

    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_labels, X_visual, X_coords = load_aligned(
        args.label_path,
        args.feature_path,
        args.community_path,
        args.geo_path,
        output_dir,
        args.feature_len,
        use_geo=use_geo,
    )

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    targets = [t for t in targets if t in df_labels.columns]
    print(f"[token_reg] targets: {targets}")

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    parameter = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "device": device,
        "feature_len": X_visual.shape[2],
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "use_geo": use_geo,
    }

    run(df_labels, X_visual, X_coords, output_dir, targets, parameter)


if __name__ == "__main__":
    main()
