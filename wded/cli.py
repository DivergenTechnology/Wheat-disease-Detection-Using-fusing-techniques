"""Command-line entry points: ``wded demo`` and ``wded run``."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PipelineConfig
from .demo import generate_sample_data
from .pipeline import run_pipeline


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="wded", description="Wheat Disease Early Detection inference pipeline"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="run the full pipeline end-to-end on synthetic data")
    demo.add_argument("--root", default="sample_data", help="where synthetic inputs are generated")
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--publish-to", default=None, help="copy dashboard artifacts to this directory")

    run = sub.add_parser("run", help="run the pipeline on a real flight bundle")
    run.add_argument("--model-dir", required=True)
    run.add_argument("--flight-dir", required=True)
    run.add_argument("--weather", default=None, help="station weather CSV")
    run.add_argument("--soil", default=None, help="soil record CSV")
    run.add_argument("--out", default="runs/latest")
    run.add_argument("--site", default="bishoftu")
    run.add_argument("--date", default="", help="capture date ISO; falls back to flight_meta.json")
    run.add_argument("--publish-to", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        generate_sample_data(args.root, seed=args.seed)
        cfg = PipelineConfig(
            model_dir=Path(args.root) / "models" / "trained",
            flight_dir=Path(args.root) / "flight",
            weather_csv=Path(args.root) / "weather.csv",
            soil_csv=Path(args.root) / "soil.csv",
            output_dir=Path(args.out),
            seed=args.seed,
        )
        summary = run_pipeline(cfg, publish_to=args.publish_to)
        print("Demo complete:", summary["tier_counts"])
        print("Outputs in:", Path(args.out).resolve())
        return 0

    cfg = PipelineConfig(
        model_dir=Path(args.model_dir),
        flight_dir=Path(args.flight_dir),
        weather_csv=Path(args.weather) if args.weather else None,
        soil_csv=Path(args.soil) if args.soil else None,
        output_dir=Path(args.out),
        site_id=args.site,
        flight_date=args.date,
    )
    summary = run_pipeline(cfg, publish_to=args.publish_to)
    print("Run complete:", summary["tier_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
