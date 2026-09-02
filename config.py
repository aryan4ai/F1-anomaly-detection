"""Central configuration for the F1 anomaly detection project."""

from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_store"
CACHE_DIR = DATA_DIR / "fastf1_cache"
PARQUET_DIR = DATA_DIR / "parquet"
LABELS_DIR = DATA_DIR / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "saved_models"

for d in [DATA_DIR, CACHE_DIR, PARQUET_DIR, LABELS_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Telemetry channels used as features
CHANNELS = ["Speed", "Throttle", "Brake", "RPM", "nGear"]

# Resampling: each lap is resampled to this many points
RESAMPLE_POINTS = 200

# Seasons to collect
DEFAULT_SEASONS = [2022, 2023, 2024]

# Train/val/test split (temporal)
TRAIN_SEASONS = [2024]
VAL_SEASON_ROUNDS = (2024, 1, 12)   # 2024 rounds 1-12
TEST_SEASON_ROUNDS = (2024, 13, 24)  # 2024 rounds 13-24

# Circuit families for circuit-aware analysis
CIRCUIT_FAMILIES = {
    "street": ["Monaco", "Singapore", "Azerbaijan", "Jeddah", "Las Vegas", "Melbourne"],
    "high_speed": ["Monza", "Spa-Francorchamps", "Silverstone", "Suzuka"],
    "mixed": ["Bahrain", "Barcelona", "Spielberg", "Zandvoort", "Lusail",
              "Interlagos", "Yas Island", "Shanghai", "Imola", "Hungaroring",
              "Mexico City", "Austin", "Montreal"],
}

# Anomaly threshold multiplier (mean + k * std)
DEFAULT_THRESHOLD_K = 2.0

# Pre-failure window: laps before a known DNF to label as "pre_failure"
PRE_FAILURE_WINDOW = 5

# Autoencoder defaults
AE_DEFAULTS = {
    "bottleneck": 16,
    "lr": 0.001,
    "epochs": 100,
    "batch_size": 16,
    "hidden_layers": [256, 64],
}

# VAE defaults
VAE_DEFAULTS = {
    "latent_dim": 16,
    "lr": 0.001,
    "epochs": 100,
    "batch_size": 16,
    "hidden_layers": [256, 64],
}

# LSTM defaults
LSTM_DEFAULTS = {
    "hidden_size": 64,
    "num_layers": 2,
    "lr": 0.001,
    "epochs": 100,
    "batch_size": 16,
}
