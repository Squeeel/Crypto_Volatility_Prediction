import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_len: int):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.x) - self.seq_len + 1

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx : idx + self.seq_len]),
            torch.from_numpy(self.y[idx + self.seq_len - 1]),
        )


class LSTM(nn.Module):
    """LSTM encoder with MLP regression head for multi-target time series."""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout_lstm: float = 0.2,
        dropout_mlp: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(64, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def make_loaders(
    train_df,
    val_df,
    test_df,
    feature_cols: list,
    target_cols: list,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> tuple:
    """Returns (train_loader, val_loader, test_loader, n_features)."""

    def _worker_init(worker_id):
        np.random.seed(seed)

    def _build(df, shuffle: bool) -> DataLoader:
        ds = TimeSeriesDataset(df[feature_cols].values, df[target_cols].values, seq_len)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            drop_last=False, worker_init_fn=_worker_init,
        )

    return (
        _build(train_df, shuffle=True),
        _build(val_df, shuffle=False),
        _build(test_df, shuffle=False),
        len(feature_cols),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            losses.append(criterion(model(x.to(device)), y.to(device)).item())
    return float(np.sqrt(np.mean(losses)))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    n_epochs: int,
    patience: int,
    device: torch.device,
    model_path: Path,
    resume: bool = True,
) -> dict:
    """
    Entraîne le modèle avec early stopping et checkpointing complet.
    Si resume=True et qu'un checkpoint existe à model_path, reprend depuis le dernier epoch sauvegardé.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    history = {"train_rmse": [], "val_rmse": []}
    best_val = float("inf")
    wait = 0
    start_epoch = 1

    checkpoint_path = Path(str(model_path).replace(".pt", "_checkpoint.pt"))

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history   = ckpt["history"]
        best_val  = ckpt["best_val"]
        wait      = ckpt["wait"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Reprise depuis l'epoch {start_epoch}  (best val={best_val:.4f})")

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_rmse = float(np.sqrt(np.mean(batch_losses)))
        val_rmse   = evaluate(model, val_loader, device)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        print(f"Epoch {epoch:3d}  train={train_rmse:.4f}  val={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            wait = 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1

        # checkpoint complet à chaque epoch
        torch.save({
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history":        history,
            "best_val":       best_val,
            "wait":           wait,
        }, checkpoint_path)

        if wait >= patience:
            print(f"Early stopping à l'epoch {epoch}  (best val={best_val:.4f})")
            break

    return history
