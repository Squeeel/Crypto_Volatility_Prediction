"""
Construit Data/CRYPTOUSDT_1m.parquet à partir des klines 1 minute CRYPTOUSDT de Binance.

Pour chaque mois manquant dans Data/CRYPTO_data/, télécharge automatiquement
le zip depuis https://data.binance.vision/ avant de construire le parquet.
"""

from datetime import date
from pathlib import Path
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

CRYPTO = "ETHUSDT"

ROOT     = Path(__file__).parent.parent
SRC_DIR  = ROOT / "Data" / f"{CRYPTO}_data"
OUT_FILE = ROOT / "Data" / f"{CRYPTO}_1m.parquet"
SRC_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL     = f"https://data.binance.vision/data/spot/monthly/klines/{CRYPTO}/1m"
START_MONTH  = date(2018, 1, 1)

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]
KEEP_COLS = [c for c in COLUMNS if c != "ignore"]
DTYPES = {
    "open":                         "float32",
    "high":                         "float32",
    "low":                          "float32",
    "close":                        "float32",
    "volume":                       "float32",
    "quote_asset_volume":           "float32",
    "number_of_trades":             "int32",
    "taker_buy_base_asset_volume":  "float32",
    "taker_buy_quote_asset_volume": "float32",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def expected_months() -> list[date]:
    """Tous les mois de START_MONTH jusqu'au dernier mois complet."""
    today = date.today()
    last  = date(today.year, today.month, 1)   # mois en cours (incomplet)
    months, cur = [], START_MONTH
    while cur < last:
        months.append(cur)
        cur = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return months


def zip_path(month: date) -> Path:
    return SRC_DIR / f"{CRYPTO}-1m-{month:%Y-%m}.zip"


def download(month: date) -> None:
    filename = f"{CRYPTO}-1m-{month:%Y-%m}.zip"
    url  = f"{BASE_URL}/{filename}"
    dest = zip_path(month)
    resp = requests.get(url, stream=True, timeout=30)
    if resp.status_code == 404:
        print(f"  [ABSENT] {filename} introuvable sur Binance Vision (mois incomplet ?)")
        return
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=filename, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS, dtype=str)
    df = df[KEEP_COLS].copy()
    df["open_time"]  = pd.to_datetime(df["open_time"].astype("int64"),  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    for col, dt in DTYPES.items():
        df[col] = df[col].astype(dt)
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    months = expected_months()
    print(f"Période cible : {months[0]:%Y-%m} → {months[-1]:%Y-%m}  ({len(months)} mois)")

    # téléchargement des mois manquants
    missing = [m for m in months if not zip_path(m).exists()]
    if missing:
        print(f"\nTéléchargement de {len(missing)} mois manquants...")
        for m in tqdm(missing, desc="Téléchargement"):
            download(m)
    else:
        print("Tous les fichiers sont déjà présents localement.")

    # lecture
    zips = sorted(p for m in months if (p := zip_path(m)).exists())
    print(f"\nLecture de {len(zips)} fichiers...")
    chunks = [read_zip(z) for z in tqdm(zips, desc="Lecture")]

    df = pd.concat(chunks, ignore_index=True)

    # nettoyage timestamps aberrants
    valid = df["open_time"].dt.year.between(2010, 2035)
    n_bad = (~valid).sum()
    if n_bad:
        print(f"[WARN] {n_bad} lignes avec timestamps invalides supprimées")
        df = df[valid]

    df = (
        df.drop_duplicates(subset="open_time")
        .sort_values("open_time")
        .set_index("open_time")
    )

    print(f"\nDataFrame final : {len(df):,} lignes × {df.shape[1]} colonnes")
    print(f"Période : {df.index[0]}  →  {df.index[-1]}")
    print(f"NaN : {df.isna().sum().sum()}")

    df.to_parquet(OUT_FILE, engine="pyarrow", compression="snappy")
    print(f"\nSauvegardé : {OUT_FILE}  ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
