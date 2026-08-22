#!/usr/bin/env python
"""CLI entry point for the network traffic anomaly detection pipeline.

Examples:
    python run.py --dataset dos_clean
    python run.py --dataset dns --tune
    python run.py --dataset both --with-autoencoder
"""
import argparse
import json

from src import config
from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["dns", "dos_clean", "both"], default="dos_clean",
        help="Which dataset to run. 'dos_clean' is the pre-cleaned DoS capture "
             "(fast, recommended default). 'both' runs dns + dos_clean.",
    )
    parser.add_argument("--tune", action="store_true",
                         help="Enable RandomizedSearchCV hyperparameter tuning for RF/XGBoost "
                              "(slower; off by default -- see README 'Fixes' item 5).")
    parser.add_argument("--with-autoencoder", action="store_true",
                         help="Also train the TensorFlow autoencoder. Off by default: it's the "
                              "weakest or tied-weakest model on both datasets, and TensorFlow is "
                              "by far the heaviest dependency in this project (see README).")
    args = parser.parse_args()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["dns", "dos_clean"] if args.dataset == "both" else [args.dataset]

    summary = {}
    for name in datasets:
        summary[name] = run(name, tune=args.tune, train_autoencoder=args.with_autoencoder)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(json.dumps(
        {d: {m: {k: round(v, 4) for k, v in met.items()} for m, met in s["metrics"].items()}
         for d, s in summary.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
