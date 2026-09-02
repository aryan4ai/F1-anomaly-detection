# F1 Telemetry Anomaly Detection

Detecting mechanical failures and anomalous behaviour in Formula 1 telemetry data using unsupervised anomaly detection. Compares five models (Autoencoder, VAE, LSTM Autoencoder, Isolation Forest, One-Class SVM) across multiple seasons, with per-channel failure mode classification and cross-circuit generalization analysis.

## Project Structure

```
f1_anomaly_detection/
├── run_pipeline.py              # CLI entry point — runs the full pipeline
├── config.py                    # Central configuration (seasons, paths, defaults)
├── requirements.txt
├── README.md
│
├── data/
│   ├── collector.py             # Downloads and preprocesses telemetry from FastF1
│   ├── labeler.py               # Generates ground truth labels from race results
│   └── splitter.py              # Temporal and circuit-aware train/test splitting
│
├── models/
│   ├── base.py                  # Abstract base class (common interface)
│   ├── autoencoder.py           # Standard dense autoencoder
│   ├── vae.py                   # Variational autoencoder
│   ├── lstm_ae.py               # LSTM-based sequence autoencoder
│   ├── isolation_forest.py      # Isolation Forest baseline
│   └── one_class_svm.py         # One-Class SVM baseline
│
├── evaluation/
│   ├── metrics.py               # Precision, recall, F1, AUC, lead time
│   └── decomposition.py         # Per-channel failure mode classification
│
├── visualization/
│   └── plots.py                 # Publication-quality figure generation
│
├── search/
│   └── hyperparameter.py        # Optuna-based hyperparameter search
│
├── data_store/                  # Created at runtime
│   ├── fastf1_cache/            # FastF1 download cache
│   ├── parquet/                 # Preprocessed telemetry
│   └── labels/                  # Ground truth labels
│
└── results/                     # Created at runtime
    ├── figures/                 # All generated plots
    ├── saved_models/            # Trained model checkpoints
    ├── model_comparison.csv
    ├── eval_results.json
    └── circuit_generalization.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Full pipeline (collect → label → train → evaluate → plot)

```bash
python run_pipeline.py --all --seasons 2022 2023 2024
```

### Individual steps

```bash
# 1. Download and preprocess telemetry
python run_pipeline.py --collect --seasons 2022 2023 2024

# 2. Generate ground truth labels
python run_pipeline.py --label --seasons 2022 2023 2024

# 3. Train all models, evaluate, generate figures
python run_pipeline.py --train --seasons 2022 2023 2024

# 4. Hyperparameter search (optional)
python run_pipeline.py --search --model autoencoder --n_trials 50

# 5. Cross-circuit generalization study
python run_pipeline.py --circuit-study --seasons 2022 2023 2024
```

## Data Sources

- **FastF1** — lap times, sector times, telemetry (speed, throttle, brake, RPM, gear), tyre compound, pit stops, race control messages. Covers 2018–present.
- All data is open source. No API keys or authentication required.

## Models

| Model | Type | Key Property |
|---|---|---|
| Autoencoder | Neural (dense) | Baseline reconstruction-based detector |
| VAE | Neural (generative) | Probabilistic reconstruction + KL divergence |
| LSTM Autoencoder | Neural (sequential) | Preserves temporal structure within laps |
| Isolation Forest | Tree ensemble | No reconstruction; scores based on isolation depth |
| One-Class SVM | Kernel method | Boundary-based; uses PCA for dimensionality reduction |

## Evaluation

- Binary classification: pre-failure laps vs everything else
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC
- Custom metric: average warning lead time (laps before known failure)
- Threshold sensitivity analysis across multiplier values
- Per-channel anomaly decomposition with failure mode classification
