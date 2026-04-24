from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR   = ROOT / "Data"
MODEL_DIR  = ROOT / "modeles"
OUTPUT_DIR = ROOT / "resultats"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS       = ["BTCUSDT", "ETHUSDT"]
N_TARGETS   = len(PAIRS)
TARGET_COLS = [f"{p}_vol_cible_bps" for p in PAIRS]

CONTEXT_LENGTH          = 252
BATCH_SIZE              = 512
HIDDEN_SIZE             = 128
DROPOUT_LSTM            = 0.2
DROPOUT_MLP             = 0.5
LEARNING_RATE           = 1e-3
EPOCHS                  = 100
EARLY_STOPPING_PATIENCE = 10

N_TRIALS        = 50
N_EPOCHS_OPTUNA = 50

SEED = 42

DEBUG = False 
