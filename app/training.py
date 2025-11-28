"""CLI to train and persist the pilot transformer."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data_utils import load_dataset
from .model import load_or_train_model, save_model, MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the internship recommendation transformer.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to the CSV dataset.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Where to save the trained model (joblib file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else None
    model_path = Path(args.model_path)

    load_or_train_model(dataset_path=dataset_path, model_path=model_path)
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
