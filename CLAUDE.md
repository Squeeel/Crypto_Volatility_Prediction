# CLAUDE.md — Lusis

## Branches
- `main` — pipeline Forex (LSTM sur 6 paires de devises, legacy)
- `crypto` — pipeline active : prédiction de volatilité BTC/ETH

## Structure
```
data_preprocessing/
  build_crypto_parquet.py      # télécharge les klines 1m depuis Binance Vision et construit {CRYPTO}_1m.parquet
  create_crypto_features.py    # BTC + ETH → {split}_features_crypto.parquet  (75 colonnes, 15m, 70/15/15)
  create_LSTM_features.ipynb   # (legacy Forex, ne pas modifier)

prediction/
  settings.py                  # config centrale — modifier PAIRS ici pour ajouter un asset
  LSTM.py                      # classe LSTM, TimeSeriesDataset, make_loaders, train_model, evaluate
  hyperparametres.ipynb        # recherche Optuna → lstm_1l.pt + lstm_1l_params.json
  inference.ipynb              # charge le modèle, génère les prédictions, métriques et graphiques

Data/
  BTCUSDT_1m.parquet           # klines 1m BTC (2018-01 → 2024-12)
  ETHUSDT_1m.parquet           # klines 1m ETH (2018-01 → 2024-12)
  {split}_features_crypto.parquet   # features ML prêtes à l'emploi (train / validation / test)

modeles/
  lstm_1l.pt                   # meilleurs poids (best val RMSE)
  lstm_1l_checkpoint.pt        # checkpoint complet (reprise si job tué)
  lstm_1l_params.json          # hidden_size et num_layers du modèle entraîné

resultats/
  optuna.db                    # études Optuna (SQLite, reprend si interrompu)
  {split}_lstm_1l_predictions.parquet
```

## Ordre d'exécution
```
build_crypto_parquet.py        # une seule fois par asset, ou pour mettre à jour
create_crypto_features.py      # régénère les parquets features
hyperparametres.ipynb          # entraînement + sélection d'hyperparamètres
inference.ipynb                # prédictions et métriques
```

## Conventions
- Tous les chemins sont relatifs à `ROOT = Path(__file__).parent.parent` — pas de chemins absolus
- Les features sont préfixées par l'asset (`BTCUSDT_*`, `ETHUSDT_*`), les cibles se terminent par `_vol_cible_bps`
- `settings.py` est la source de vérité pour `PAIRS`, `TARGET_COLS`, `CONTEXT_LENGTH`, etc.
- `DEBUG = True` dans `settings.py` pour charger 1 % des données et 3 epochs (itération rapide)
- `DEBUG = False` avant de lancer en prod

## Environnement
- Python 3.12, venv dans `env/` — utiliser `env/bin/python` et `env/bin/pip`
- GPU CUDA disponible (torch 2.10 + drivers NVIDIA)
- Exécution longue sur SSH : `tmux` + `jupyter nbconvert --to notebook --execute --inplace`

## Résilience entraînement
- Optuna : `load_if_exists=True` sur SQLite → reprend les trials déjà complétés
- `train_model` : checkpoint complet sauvegardé à chaque epoch (`lstm_1l_checkpoint.pt`) → perd au plus 1 epoch si job tué

## Ajouter un nouvel asset
1. Ajouter le ticker dans `build_crypto_parquet.py` (`CRYPTO = "XYZUSDT"`) et l'exécuter
2. Ajouter le ticker dans `CRYPTOS` de `create_crypto_features.py` et l'exécuter
3. Ajouter le ticker dans `PAIRS` de `prediction/settings.py`
