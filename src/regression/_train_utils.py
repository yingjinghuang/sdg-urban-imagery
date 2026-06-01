"""Training-loop helpers shared by regression entry scripts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class AverageMeter:
    """Running mean tracker."""

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Stop training when val loss stops improving for ``patience`` epochs."""

    def __init__(self, patience: int = 7, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, save_path: str) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), save_path)
            self.counter = 0


def adjust_learning_rate(optimizer, epoch: float, parameter: dict) -> float:
    """Cosine schedule with linear warmup."""
    warmup = parameter["warmup_epochs"]
    if epoch < warmup:
        lr = parameter["lr"] * epoch / warmup
    else:
        progress = (epoch - warmup) / (parameter["num_epochs"] - warmup)
        lr = parameter["lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, epoch: int, parameter: dict) -> None:
    model.train()
    use_geo = parameter.get("use_geo", True)
    for i, batch in enumerate(loader):
        adjust_learning_rate(optimizer, epoch + i / len(loader), parameter)
        features, coords, labels = batch
        features = features.to(parameter["device"])
        labels = labels.to(parameter["device"])
        if use_geo:
            coords = coords.to(parameter["device"])
            pred = model(features, coords)
        else:
            pred = model(features)
        optimizer.zero_grad()
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, criterion, parameter: dict) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses = AverageMeter("val_loss")
    y_true, y_pred = [], []
    use_geo = parameter.get("use_geo", True)
    with torch.no_grad():
        for features, coords, labels in loader:
            features = features.to(parameter["device"])
            labels = labels.to(parameter["device"])
            if use_geo:
                coords = coords.to(parameter["device"])
                out = model(features, coords)
            else:
                out = model(features)
            losses.update(criterion(out, labels).item(), features.size(0))
            y_true.append(labels.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    return np.vstack(y_true), np.vstack(y_pred), losses.avg


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    targets: list[str],
) -> pd.DataFrame:
    """MAE / MSE / R² per target, plus the "all" combined-with-train version.

    ``all_*`` columns are the metrics computed after concatenating the
    train-set labels (kept perfect) with the test-set predictions. This
    is the "citywide reconstruction R²" the paper reports.
    """

    def _eval(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float, float]:
        mask = ~np.isnan(yt)
        if mask.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        yt, yp = yt[mask], yp[mask]
        return mean_absolute_error(yt, yp), mean_squared_error(yt, yp), r2_score(yt, yp)

    rows = []
    for i, target in enumerate(targets):
        mae, mse, r2 = _eval(y_true[:, i], y_pred[:, i])
        all_mae, all_mse, all_r2 = _eval(
            np.concatenate([y_train[:, i], y_true[:, i]]),
            np.concatenate([y_train[:, i], y_pred[:, i]]),
        )
        rows.append([target, mae, mse, r2, all_mae, all_mse, all_r2])
    return pd.DataFrame(
        rows,
        columns=["target", "mae", "mse", "r2", "all_mae", "all_mse", "all_r2"],
    )
