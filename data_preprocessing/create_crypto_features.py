"""
Crée Data/{train,validation,test}_features_LSTM.parquet depuis les parquets 1m.

Features par asset (BTC, ETH) :
  0 - base          : log_range, log_return, cible
  1 - rolling       : vol réalisée + momentum (5m, 15m, 1h, 4h, 1j)
  2 - vol structure : ratios EWM court/long, momentum de vol
  3 - risque        : kurtosis, skew, asymétrie down/up
  4 - prix          : z-scores rendements, momentum prix
  5 - volume/flux   : taker buy ratio, volume/trades z-scores, MFI
  6 - micro         : position dans le range journalier, rang de vol relatif

Features croisées BTC/ETH :
  7 - corrélation rolling des rendements
  8 - ratio de volatilité et spread de term structure

Features temporelles (partagées) :
  9 - cycliques intraday/hebdo, flag weekend
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "Data"

CRYPTOS = ["BTCUSDT", "ETHUSDT"]
H       = 240   # horizon cible (minutes)
FREQ    = 15    # sous-échantillonnage final (minutes)
EPSILON = 1e-9

SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}

TARGET_COLS = [f"{c}_vol_cible_bps" for c in CRYPTOS]


# ── helpers ───────────────────────────────────────────────────────────────────

def load(crypto: str) -> pd.DataFrame:
    path = DATA_DIR / f"{crypto}_1m.parquet"
    df   = pd.read_parquet(path).sort_index()
    start = df.index[0].ceil("h")
    end   = df.index[-1].floor("h") - pd.Timedelta(minutes=1)
    return df.loc[start:end]


def per_asset_features(df: pd.DataFrame, crypto: str, feat: pd.DataFrame) -> None:
    """Calcule les features d'un seul asset et les écrit dans feat (in-place)."""
    p = crypto  # préfixe

    high   = np.log(df["high"])
    low    = np.log(df["low"])
    close  = np.log(df["close"])
    volume = df["volume"]
    quote_vol = df["quote_asset_volume"]
    n_trades  = df["number_of_trades"]
    taker_buy = df["taker_buy_quote_asset_volume"]

    log_range  = high - low
    log_return = close.diff()

    # Groupe 0 : cible + base
    feat[f"{p}_vol_cible_bps"] = log_range.rolling(H, min_periods=H).mean().shift(-H) * 10_000
    feat[f"{p}_log_range"]     = log_range
    feat[f"{p}_log_return"]    = log_return

    # Groupe 1 : agrégations rolling
    for w in [5, 15, 60, 240, 1440]:
        feat[f"{p}_realized_vol_{w}m"] = log_range.rolling(w).sum()
        feat[f"{p}_momentum_{w}m"]     = log_return.rolling(w).sum()

    # Groupe 2 : structure de volatilité
    ewm60   = log_range.ewm(span=60,   adjust=False).mean()
    ewm240  = log_range.ewm(span=240,  adjust=False).mean()
    ewm1440 = log_range.ewm(span=1440, adjust=False).mean()

    feat[f"{p}_vol_term_structure_1h_4h"] = ewm60  / (ewm240  + EPSILON) - 1
    feat[f"{p}_vol_term_structure_4h_1d"] = ewm240 / (ewm1440 + EPSILON) - 1
    feat[f"{p}_vol_momentum_4h"]          = ewm240  / (ewm240.shift(240)   + EPSILON) - 1
    feat[f"{p}_vol_momentum_1d"]          = ewm1440 / (ewm1440.shift(1440) + EPSILON) - 1

    # Groupe 3 : risque
    feat[f"{p}_kurtosis_returns_4h"] = log_return.rolling(240).kurt()
    feat[f"{p}_skewness_returns_4h"] = log_return.rolling(240).skew()
    vol_down = np.sqrt((log_return.where(log_return < 0, 0) ** 2).rolling(240).mean())
    vol_up   = np.sqrt((log_return.where(log_return > 0, 0) ** 2).rolling(240).mean())
    feat[f"{p}_vol_asymmetry_4h"] = vol_down / (vol_up + EPSILON) - 1

    # Groupe 4 : dynamique de prix
    feat[f"{p}_return_zscore_1h"] = (
        (log_return - log_return.rolling(60).mean())
        / (log_return.rolling(60).std() + EPSILON)
    )
    feat[f"{p}_return_zscore_4h"] = (
        (log_return - log_return.rolling(240).mean())
        / (log_return.rolling(240).std() + EPSILON)
    )
    feat[f"{p}_price_momentum_4h"] = close.diff(240)
    feat[f"{p}_price_momentum_1d"] = close.diff(1440)

    # Groupe 5 : volume / flux
    taker_ratio = taker_buy / (quote_vol + EPSILON)
    feat[f"{p}_taker_buy_ratio"]      = taker_ratio * 2 - 1
    feat[f"{p}_taker_buy_ratio_ma60"] = taker_ratio.rolling(60).mean() * 2 - 1
    feat[f"{p}_volume_zscore_4h"] = (
        (volume - volume.rolling(240).mean()) / (volume.rolling(240).std() + EPSILON)
    )
    feat[f"{p}_volume_zscore_1d"] = (
        (volume - volume.rolling(1440).mean()) / (volume.rolling(1440).std() + EPSILON)
    )
    feat[f"{p}_trades_zscore_4h"] = (
        (n_trades - n_trades.rolling(240).mean()) / (n_trades.rolling(240).std() + EPSILON)
    )
    feat[f"{p}_log_n_trades"] = np.log1p(n_trades)
    mfi = ta.mfi(high=df["high"], low=df["low"], close=df["close"], volume=volume, length=14)
    feat[f"{p}_mfi_centered"] = mfi / 50.0 - 1.0

    # Groupe 6 : microstructure
    price  = df["close"]
    min_1d = price.rolling(1440).min()
    max_1d = price.rolling(1440).max()
    feat[f"{p}_position_range_1d"] = 2 * ((price - min_1d) / (max_1d - min_1d + EPSILON)) - 1


def cross_asset_features(dfs: dict[str, pd.DataFrame], feat: pd.DataFrame) -> None:
    """Features croisées BTC/ETH."""
    btc_ret = np.log(dfs["BTCUSDT"]["close"]).diff()
    eth_ret = np.log(dfs["ETHUSDT"]["close"]).diff()
    btc_range = np.log(dfs["BTCUSDT"]["high"]) - np.log(dfs["BTCUSDT"]["low"])
    eth_range = np.log(dfs["ETHUSDT"]["high"]) - np.log(dfs["ETHUSDT"]["low"])

    btc_ewm240 = btc_range.ewm(span=240, adjust=False).mean()
    eth_ewm240 = eth_range.ewm(span=240, adjust=False).mean()

    # Ratio de volatilité réalisée
    feat["btceth_vol_ratio_4h"] = btc_ewm240 / (eth_ewm240 + EPSILON) - 1

    # Rang de vol relative [-1, 1] : +1 → BTC plus volatile, -1 → ETH plus volatile
    vol_df = pd.DataFrame({"BTC": btc_ewm240, "ETH": eth_ewm240})
    ranks  = vol_df.rank(axis=1, method="first")
    feat["btceth_vol_rank_btc"] = 2 * ((ranks["BTC"] - 1) / 1) - 1  # 2 assets → rang ∈ {0,1}

    # Corrélation rolling des rendements (240m et 1440m)
    feat["btceth_return_corr_4h"] = btc_ret.rolling(240).corr(eth_ret)
    feat["btceth_return_corr_1d"] = btc_ret.rolling(1440).corr(eth_ret)

    # Lead-lag : rendement BTC décalé d'1 minute prédit-il ETH ?
    feat["btceth_leadlag_1m"] = btc_ret.shift(1).rolling(60).corr(eth_ret)

    # Spread de term structure
    btc_ts = (btc_range.ewm(span=60, adjust=False).mean()
              / (btc_range.ewm(span=240, adjust=False).mean() + EPSILON) - 1)
    eth_ts = (eth_range.ewm(span=60, adjust=False).mean()
              / (eth_range.ewm(span=240, adjust=False).mean() + EPSILON) - 1)
    feat["btceth_termstructure_spread"] = btc_ts - eth_ts


def temporal_features(feat: pd.DataFrame) -> None:
    ts = feat.index
    feat["minute_of_day_sin"] = np.sin(2 * np.pi * (ts.hour * 60 + ts.minute) / 1440)
    feat["minute_of_day_cos"] = np.cos(2 * np.pi * (ts.hour * 60 + ts.minute) / 1440)
    feat["day_of_week_sin"]   = np.sin(2 * np.pi * ts.dayofweek / 7)
    feat["day_of_week_cos"]   = np.cos(2 * np.pi * ts.dayofweek / 7)
    feat["is_weekend"]        = (ts.dayofweek >= 5).astype("int8") * 2 - 1


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    dfs = {}
    for crypto in CRYPTOS:
        dfs[crypto] = load(crypto)
        print(f"{crypto}: {dfs[crypto].shape}  {dfs[crypto].index[0].date()} → {dfs[crypto].index[-1].date()}")

    # Aligne sur l'intersection des index
    common_idx = dfs[CRYPTOS[0]].index
    for crypto in CRYPTOS[1:]:
        common_idx = common_idx.intersection(dfs[crypto].index)
    dfs = {c: df.loc[common_idx] for c, df in dfs.items()}
    print(f"\nIndex commun : {len(common_idx):,} minutes")

    feat = pd.DataFrame(index=common_idx)

    for crypto in CRYPTOS:
        print(f"Features {crypto}...")
        per_asset_features(dfs[crypto], crypto, feat)

    print("Features croisées BTC/ETH...")
    cross_asset_features(dfs, feat)

    print("Features temporelles...")
    temporal_features(feat)

    del dfs
    gc.collect()

    # Nettoyage & sous-échantillonnage
    feat.dropna(inplace=True)
    feat_15m = feat[feat.index.minute % FREQ == 0].copy()

    # Cibles en premier, reste trié
    other_cols = sorted(c for c in feat_15m.columns if c not in TARGET_COLS)
    feat_15m   = feat_15m[TARGET_COLS + other_cols]
    print(f"\nShape finale (15m) : {feat_15m.shape}")

    # Split temporel 70 / 15 / 15
    n       = len(feat_15m)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["validation"])
    splits  = {
        "train":      feat_15m.iloc[:n_train],
        "validation": feat_15m.iloc[n_train : n_train + n_val],
        "test":       feat_15m.iloc[n_train + n_val :],
    }

    print()
    for name, split in splits.items():
        out = DATA_DIR / f"{name}_features_crypto.parquet"
        split.to_parquet(out, engine="pyarrow", compression="snappy")
        print(
            f"{name:12s} {split.shape}  "
            f"{split.index[0].date()} → {split.index[-1].date()}  "
            f"→ {out.name}"
        )


if __name__ == "__main__":
    main()
