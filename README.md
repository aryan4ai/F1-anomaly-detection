<img width="678" height="452" alt="images" src="https://github.com/user-attachments/assets/18273d3f-4ab3-418f-8b36-0dfd1c7fc35a" />




# F1 Telemetry Anomaly Detection

Detecting mechanical failures and anomalous behaviour in Formula 1 telemetry data using unsupervised anomaly detection. Compares five models (Autoencoder, VAE, LSTM Autoencoder, Isolation Forest, One-Class SVM) across multiple seasons, with per-channel failure mode classification and cross-circuit generalization analysis.



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
 1. Download and preprocess telemetry

 2. Generate ground truth labels

 3. Train all models, evaluate, generate figures

 4. Hyperparameter search (optional)

 5. Cross-circuit generalization study

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
