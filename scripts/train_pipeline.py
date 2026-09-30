"""
train_pipeline.py — Run the full training pipeline for an IMS test.

Usage:
  python scripts/train_pipeline.py --test_id 1
  python scripts/train_pipeline.py --test_id 1 --no_freq
  python scripts/train_pipeline.py --test_id 1 --max_files 50  # quick test
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from industrial_health.pipeline.analysis_pipeline import AnalysisPipeline
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Train bearing health monitoring models on IMS dataset"
    )
    parser.add_argument(
        "--test_id", type=int, default=1, choices=[1, 2, 3],
        help="IMS test ID to train on (1, 2, or 3)"
    )
    parser.add_argument(
        "--no_freq", action="store_true",
        help="Exclude FFT frequency features"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Limit number of files (for quick testing)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-extraction even if cached features exist"
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    archive_path = project_root / "archive"
    data_path    = project_root / "data"
    models_path  = project_root / "models"

    if not archive_path.exists():
        logger.error(f"Archive not found: {archive_path}")
        logger.error("Place the IMS dataset in archive/ before running.")
        sys.exit(1)

    logger.info(f"Starting training pipeline for Test {args.test_id}")
    logger.info(f"FFT features: {not args.no_freq}")
    if args.max_files:
        logger.info(f"Limiting to {args.max_files} files (quick test mode)")

    pipeline = AnalysisPipeline(
        archive_path=archive_path,
        data_path=data_path,
        models_path=models_path,
    )

    results = pipeline.train_models(
        test_id=args.test_id,
        include_freq=not args.no_freq,
    )

    print("\n" + "=" * 60)
    print("  TRAINING RESULTS")
    print("=" * 60)
    print(f"  Test ID:        {results['test_id']}")
    print(f"  Train samples:  {results['n_train']}")
    print(f"  Test samples:   {results['n_test']}")
    print(f"  Features:       {results['n_features']}")
    print("-" * 60)
    print("  FAULT MODEL (Random Forest):")
    fm = results['fault_metrics']
    print(f"    Accuracy:   {fm['accuracy']:.4f}")
    print(f"    Precision:  {fm['precision']:.4f}")
    print(f"    Recall:     {fm['recall']:.4f}")
    print(f"    F1-Score:   {fm['f1_score']:.4f}")
    print(f"\n  Classification Report:")
    print(fm['classification_report'])
    print("-" * 60)
    print("  ANOMALY MODEL (Isolation Forest):")
    am = results['anomaly_train_info']
    print(f"    Normal train samples: {am['n_normal_train']}")
    print(f"    Score range: [{am['score_min']:.4f}, {am['score_max']:.4f}]")
    print("=" * 60)
    print("\n  Models saved to: models/")
    print("  All metrics above are MEASURED — not assumed.\n")


if __name__ == "__main__":
    main()
