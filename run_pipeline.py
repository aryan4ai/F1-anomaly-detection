#!/usr/bin/env python3
"""
F1 Telemetry Anomaly Detection — Full Pipeline

Usage:
    python run_pipeline.py --collect --seasons 2022 2023 2024
    python run_pipeline.py --train --evaluate
    python run_pipeline.py --all --seasons 2022 2023 2024
    python run_pipeline.py --search --model autoencoder --n_trials 30
    python run_pipeline.py --circuit-study

Steps:
    1. collect:        Download and preprocess telemetry → parquet
    2. label:          Generate ground truth labels from race results
    3. train:          Train all models on training split (normal laps only)
    4. evaluate:       Evaluate all models on test split, generate comparison
    5. decompose:      Per-channel anomaly decomposition on test set
    6. search:         Hyperparameter search (optional, slow)
    7. circuit-study:  Leave-one-circuit-out generalization analysis
    8. plot:           Generate all figures

    --all runs 1-5 and 8.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_SEASONS,
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)
from data import TelemetryCollector, GroundTruthLabeler, DataSplitter
from models import (
    AutoencoderDetector,
    VAEDetector,
    LSTMAutoencoderDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
)
from evaluation import Evaluator, AnomalyDecomposer
from visualization import PlotGenerator
from search import HyperparameterSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def get_all_models() -> list:
    """Instantiate all models with default hyperparameters."""
    return [
        AutoencoderDetector(),
        VAEDetector(),
        LSTMAutoencoderDetector(),
        IsolationForestDetector(),
        OneClassSVMDetector(),
    ]


def step_collect(seasons: list[int], force: bool = False) -> pd.DataFrame:
    logger.info("=== STEP 1: Collecting telemetry for seasons %s ===", seasons)
    collector = TelemetryCollector(seasons=seasons)
    meta = collector.collect_all(force=force)
    logger.info("Collected %d laps total.", len(meta))
    return meta


def step_label(seasons: list[int]) -> pd.DataFrame:
    logger.info("=== STEP 2: Generating ground truth labels ===")
    labeler = GroundTruthLabeler()
    labels = labeler.label_all(seasons)
    counts = labels["Label"].value_counts()
    logger.info("Label distribution:\n%s", counts.to_string())
    return labels


def step_train_and_evaluate(seasons: list[int]):
    logger.info("=== STEP 3-5: Train, evaluate, decompose ===")

    collector = TelemetryCollector(seasons=seasons)
    X, meta = collector.load_laps_matrix()
    if X.shape[0] == 0:
        logger.error("No data loaded. Run --collect first.")
        return

    # Merge labels
    labeler = GroundTruthLabeler()
    meta = labeler.merge_with_metadata(meta)
    logger.info("Data shape: %s, Labels: %s",
                X.shape, meta["Label"].value_counts().to_dict())

    # Split
    splitter = DataSplitter()
    splits = splitter.temporal_split(X, meta)

    X_train_all, meta_train = splits["train"]
    X_val, meta_val = splits["val"]
    X_test, meta_test = splits["test"]

    # Train only on normal laps
    X_train, meta_train_normal = splitter.get_normal_only(X_train_all, meta_train)
    logger.info("Train (normal only): %d laps | Val: %d | Test: %d",
                len(X_train), len(X_val), len(X_test))

    if len(X_train) == 0:
        logger.error("No training data after filtering. Check season/round config.")
        return

    models = get_all_models()
    evaluator = Evaluator()
    plotter = PlotGenerator()
    decomposer = AnomalyDecomposer()

    eval_results = []
    for model in models:
        logger.info("Training %s ...", model.name)
        try:
            model.fit(X_train)
            model.save(MODELS_DIR / model.name)

            # Plot training loss if available
            if hasattr(model, "train_losses_") and model.train_losses_:
                plotter.plot_training_loss(model.train_losses_, model.name)

            # Evaluate on test set
            result = evaluator.evaluate(model, X_test, meta_test)
            eval_results.append(result)
            logger.info(
                "%s — P=%.3f R=%.3f F1=%.3f AUC=%s Lead=%.1f laps",
                model.name,
                result["precision"],
                result["recall"],
                result["f1"],
                f"{result['roc_auc']:.3f}" if result["roc_auc"] else "N/A",
                result["avg_warning_lead_time"] or 0,
            )

            # Plots
            scores = model.score(X_test)
            threshold = getattr(model, "threshold_", 0)
            plotter.plot_anomaly_scores(scores, meta_test, threshold, model.name)

            if result["confusion_matrix"]:
                plotter.plot_confusion_matrix(result["confusion_matrix"], model.name)

            # Threshold sensitivity
            sens_df = evaluator.threshold_sensitivity(model, X_test, meta_test)
            sens_df.to_csv(RESULTS_DIR / f"threshold_sensitivity_{model.name}.csv", index=False)
            plotter.plot_threshold_sensitivity(sens_df, model.name)

            # Channel errors and decomposition
            ch_errs = model.channel_errors(X_test)
            plotter.plot_channel_error_heatmap(ch_errs, meta_test, model_name=model.name)

            decomp = decomposer.decompose(model, X_test, meta_test)
            if not decomp.empty:
                decomp.to_csv(RESULTS_DIR / f"decomposition_{model.name}.csv", index=False)
                fm_summary = decomposer.failure_mode_summary(decomp)
                plotter.plot_failure_modes(fm_summary)
                ch_imp = decomposer.channel_importance(decomp)
                ch_imp.to_csv(RESULTS_DIR / f"channel_importance_{model.name}.csv", index=False)

            # Telemetry comparison plot (first anomalous vs first normal)
            preds = model.predict(X_test)
            anom_indices = np.where(preds == 1)[0]
            norm_indices = np.where(preds == 0)[0]
            if len(anom_indices) > 0 and len(norm_indices) > 0:
                plotter.plot_telemetry_comparison(
                    X_test, meta_test,
                    normal_idx=int(norm_indices[0]),
                    anomaly_idx=int(anom_indices[0]),
                    model_name=model.name,
                )

        except Exception as e:
            logger.error("Failed on %s: %s", model.name, e, exc_info=True)

    # Model comparison
    if eval_results:
        comparison = evaluator.compare_models(models, X_test, meta_test)
        comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        plotter.plot_model_comparison(comparison)
        plotter.plot_roc_curves(eval_results)
        plotter.plot_pr_curves(eval_results)
        logger.info("Model comparison:\n%s", comparison.to_string(index=False))

    # Save all eval results as JSON
    serializable = []
    for r in eval_results:
        s = {k: v for k, v in r.items() if k not in ("roc_curve", "pr_curve")}
        serializable.append(s)
    with open(RESULTS_DIR / "eval_results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info("All results saved to %s", RESULTS_DIR)
    logger.info("All figures saved to %s", FIGURES_DIR)


def step_search(model_name: str, seasons: list[int], n_trials: int):
    logger.info("=== Hyperparameter search for %s (%d trials) ===", model_name, n_trials)

    collector = TelemetryCollector(seasons=seasons)
    X, meta = collector.load_laps_matrix()
    labeler = GroundTruthLabeler()
    meta = labeler.merge_with_metadata(meta)

    splitter = DataSplitter()
    splits = splitter.temporal_split(X, meta)
    X_train_all, meta_train = splits["train"]
    X_val, meta_val = splits["val"]
    X_train, _ = splitter.get_normal_only(X_train_all, meta_train)

    searcher = HyperparameterSearch()
    plotter = PlotGenerator()

    if model_name == "autoencoder":
        best, log = searcher.search_autoencoder(X_train, X_val, meta_val, n_trials)
        for param in ["bottleneck", "lr"]:
            if param in log.columns:
                plotter.plot_hp_search(log, param, model_name)
    elif model_name == "vae":
        best, log = searcher.search_vae(X_train, X_val, meta_val, n_trials)
    elif model_name == "lstm":
        best, log = searcher.search_lstm(X_train, X_val, meta_val, n_trials)
    else:
        logger.error("Unknown model: %s", model_name)
        return

    logger.info("Best params: %s", best)


def step_circuit_study(seasons: list[int]):
    logger.info("=== Leave-one-circuit-out generalization study ===")

    collector = TelemetryCollector(seasons=seasons)
    X, meta = collector.load_laps_matrix()
    labeler = GroundTruthLabeler()
    meta = labeler.merge_with_metadata(meta)

    circuits = meta["Circuit"].unique()
    splitter = DataSplitter()
    evaluator = Evaluator()
    plotter = PlotGenerator()

    rows = []
    for circuit in circuits:
        logger.info("Testing on %s ...", circuit)
        try:
            splits = splitter.leave_one_circuit_out(X, meta, circuit)
            X_train_all, meta_train = splits["train"]
            X_test, meta_test = splits["test"]
            X_train, _ = splitter.get_normal_only(X_train_all, meta_train)

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            model = AutoencoderDetector(epochs=50)
            model.fit(X_train)
            result = evaluator.evaluate(model, X_test, meta_test)
            rows.append({
                "circuit": circuit,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "n_test": len(X_test),
                "n_positive": result["n_positive"],
            })
        except Exception as e:
            logger.warning("Circuit %s failed: %s", circuit, e)

    if rows:
        circuit_df = pd.DataFrame(rows)
        circuit_df.to_csv(RESULTS_DIR / "circuit_generalization.csv", index=False)
        plotter.plot_circuit_performance(circuit_df)
        logger.info("Circuit results:\n%s", circuit_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="F1 Telemetry Anomaly Detection Pipeline")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--collect", action="store_true", help="Step 1: collect telemetry")
    parser.add_argument("--label", action="store_true", help="Step 2: generate labels")
    parser.add_argument("--train", action="store_true", help="Steps 3-5: train + evaluate")
    parser.add_argument("--evaluate", action="store_true", help="Alias for --train")
    parser.add_argument("--search", action="store_true", help="Hyperparameter search")
    parser.add_argument("--model", type=str, default="autoencoder", help="Model for HP search")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--circuit-study", action="store_true", help="Circuit generalization")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    if args.all:
        step_collect(args.seasons, force=args.force)
        step_label(args.seasons)
        step_train_and_evaluate(args.seasons)
        return

    if args.collect:
        step_collect(args.seasons, force=args.force)
    if args.label:
        step_label(args.seasons)
    if args.train or args.evaluate:
        step_train_and_evaluate(args.seasons)
    if args.search:
        step_search(args.model, args.seasons, args.n_trials)
    if args.circuit_study:
        step_circuit_study(args.seasons)

    if not any([args.collect, args.label, args.train, args.evaluate,
                args.search, args.circuit_study, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
