"""
Entraînement rapide + inférence sur les données crypto (BTC/ETH).
Pas d'Optuna — hyperparamètres fixes issus de settings.py.
Lance depuis prediction/ : python quickrun.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from LSTM import LSTM, make_loaders, train_model, evaluate, set_seed
from settings import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    PAIRS, N_TARGETS, TARGET_COLS,
    CONTEXT_LENGTH, BATCH_SIZE,
    HIDDEN_SIZE, DROPOUT_LSTM, DROPOUT_MLP, LEARNING_RATE,
    EPOCHS, EARLY_STOPPING_PATIENCE, SEED,
)

MODEL_PATH      = MODEL_DIR / "lstm_quickrun.pt"
PREDICTIONS_DIR = OUTPUT_DIR

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ── Données ───────────────────────────────────────────────────────────────────

splits = {}
for name in ("train", "validation", "test"):
    path = DATA_DIR / f"{name}_features_crypto.parquet"
    splits[name] = pd.read_parquet(path).ffill()
    print(f"{name:12s} {splits[name].shape}  {splits[name].index[0].date()} → {splits[name].index[-1].date()}")

train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
feature_cols = [c for c in train_df.columns if c not in TARGET_COLS]
print(f"\nFeatures : {len(feature_cols)}  |  Cibles : {N_TARGETS}")

train_loader, val_loader, test_loader, n_features = make_loaders(
    train_df, val_df, test_df, feature_cols, TARGET_COLS,
    seq_len=CONTEXT_LENGTH, batch_size=BATCH_SIZE, seed=SEED,
)

# ── Entraînement ──────────────────────────────────────────────────────────────

model = LSTM(
    n_features=n_features,
    n_targets=N_TARGETS,
    hidden_size=HIDDEN_SIZE,
    num_layers=1,
    dropout_lstm=DROPOUT_LSTM,
    dropout_mlp=DROPOUT_MLP,
)
print(f"\n{model}\n")

train_model(
    model, train_loader, val_loader,
    lr=LEARNING_RATE, n_epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    device=device, model_path=MODEL_PATH,
)

# ── Inférence ─────────────────────────────────────────────────────────────────

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

def run_inference(loader):
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)

pred_cols = [f"{p}_pred_bps" for p in PAIRS]
loaders   = {"train": (train_loader, train_df), "validation": (val_loader, val_df), "test": (test_loader, test_df)}
all_preds = {}

for split_name, (loader, ref_df) in loaders.items():
    preds_np = run_inference(loader)
    idx      = ref_df.index[CONTEXT_LENGTH - 1:]
    all_preds[split_name] = pd.DataFrame(preds_np, index=idx, columns=pred_cols)

# ── Métriques ─────────────────────────────────────────────────────────────────

rows = []
for split_name, preds_df in all_preds.items():
    ref_df = loaders[split_name][1]
    tgt    = ref_df[TARGET_COLS].iloc[CONTEXT_LENGTH - 1:].values
    pred   = preds_df.values
    for pair, rmse in zip(PAIRS, np.sqrt(np.mean((pred - tgt) ** 2, axis=0))):
        rows.append({"split": split_name, "pair": pair, "RMSE (BPS)": round(float(rmse), 4)})

metrics = pd.DataFrame(rows).pivot(index="pair", columns="split", values="RMSE (BPS)")
metrics.loc["MOYENNE"] = metrics.mean()
print("\n── Résultats ────────────────────────────────────────────────────────────")
print(metrics.to_string())

# ── Graphique ─────────────────────────────────────────────────────────────────

MAX_POINTS = 3000
COLORS = {"train": "#2196F3", "validation": "#4CAF50", "test": "#FF5722"}

fig, axes = plt.subplots(N_TARGETS, 1, figsize=(16, 5 * N_TARGETS), sharex=False)
if N_TARGETS == 1:
    axes = [axes]
fig.suptitle("Prédictions vs Cibles — LSTM quickrun", fontsize=13)

for ax, pair, target_col, pred_col in zip(axes, PAIRS, TARGET_COLS, pred_cols):
    for split_name, preds_df in all_preds.items():
        ref_df  = loaders[split_name][1]
        tgt     = ref_df[target_col].iloc[CONTEXT_LENGTH - 1:]
        step    = max(1, len(preds_df) // MAX_POINTS)
        color   = COLORS[split_name]
        ax.plot(preds_df.index[::step], preds_df[pred_col][::step],
                color=color, lw=0.9, alpha=0.85, label=f"Prédiction ({split_name})")
        ax.plot(tgt.index[::step], tgt.values[::step],
                color=color, lw=0.9, alpha=0.35, ls="--", label=f"Réel ({split_name})")
    ax.set_title(pair)
    ax.set_ylabel("Volatilité (BPS)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)

plt.tight_layout()
fig_path = OUTPUT_DIR / "quickrun_predictions.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
print(f"\nGraphique : {fig_path}")
plt.show()
