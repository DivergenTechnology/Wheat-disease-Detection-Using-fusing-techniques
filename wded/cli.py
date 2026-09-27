"""Command-line entry points: ``wded demo``, ``wded run`` and ``wded report``."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from .config import PipelineConfig
from .demo import generate_sample_data
from .pipeline import run_pipeline
from .report import generate_field_report


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
    demo.add_argument("--format", choices=["npz", "geotiff"], default="npz",
                      help="synthetic flight format (geotiff exercises the real raster reader)")
    demo.add_argument("--date", default=None, dest="capture_date",
                      help="capture date ISO (overrides the default flight date)")
    demo.add_argument("--hotspot-scale", type=float, default=1.0,
                      help="scales disease hotspot intensity (multi-date scenarios)")
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

    report = sub.add_parser("report", help="render a printable HTML field risk report from a run")
    report.add_argument("--run", default="runs/latest", help="pipeline output directory")
    report.add_argument("--out", default=None, help="report path (default <run>/field_report.html)")

    trend = sub.add_parser("trend", help="build a multi-date risk trend from several run dirs")
    trend.add_argument("--runs", nargs="+", required=True,
                       help="pipeline output directories, one per flight date")
    trend.add_argument("--out", default="runs/trend/trend.json")
    trend.add_argument("--publish-to", default=None, help="copy trend.json to this directory")

    minfo = sub.add_parser(
        "model-info",
        help="inspect a local model directory and verify the trained SSCNN integrates",
    )
    minfo.add_argument("model_dir", help="directory holding the trained checkpoint")
    minfo.add_argument("--device", default=None, help="force device (cpu/cuda); default: metadata.json or cpu")
    minfo.add_argument("--architecture", default=None,
                       help="import path 'pkg.module:ClassName' for state_dict-only checkpoints")
    minfo.add_argument("--no-validate", action="store_true", help="skip the forward-contract dry run")

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        generate_sample_data(
            args.root,
            seed=args.seed,
            fmt=args.format,
            capture_date=args.capture_date,
            hotspot_scale=args.hotspot_scale,
        )
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

    if args.cmd == "trend":
        from .trend import build_trend, write_trend

        trend_data = build_trend(args.runs)
        out = write_trend(trend_data, args.out)
        if args.publish_to:
            pub = Path(args.publish_to)
            pub.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, pub / "trend.json")
            print("Published trend.json to", (pub / "trend.json").resolve())
        counts = trend_data["tile_trend_counts"]
        print(
            f"Trend over {trend_data['n_dates']} flights "
            f"({trend_data['first_date']} -> {trend_data['last_date']}): "
            f"field {trend_data['field_trend']} ({trend_data['field_delta']:+.3f}); "
            f"tiles worsening={counts['worsening']} stable={counts['stable']} "
            f"improving={counts['improving']} new={counts['new']}"
        )
        print("Trend written:", out.resolve())
        return 0

    if args.cmd == "report":
        out = generate_field_report(args.run, args.out)
        print("Field report written:", out.resolve())
        return 0

    if args.cmd == "model-info":
        from .models import describe_model_dir

        report = describe_model_dir(
            args.model_dir,
            device=args.device,
            architecture=args.architecture,
            validate=not args.no_validate,
        )
        print("WDED model directory report")
        print(f"  model_dir     : {report['model_dir']}")
        print(f"  exists        : {report['exists']}")
        print(f"  checkpoint    : {report['checkpoint']}")
        print(f"  format        : {report['checkpoint_format']}")
        print(f"  architecture  : {report['architecture']}")
        print(f"  parameters    : {report['n_parameters']}")
        print(f"  device        : {report['device']}")
        print(f"  temperature   : {report['temperature']}")
        print(f"  gradcam_layer : {report['gradcam_layer']}")
        print(f"  dry-run       : {'passed' if report['validated'] else 'skipped'}")
        if report["ok"]:
            print("\nModel integrated successfully. Run the pipeline with:")
            print(f"  wded run --model-dir {args.model_dir} --flight-dir <flight_dir>")
            return 0
        print(f"\nFAILED: {report['error']}")
        print("See the README 'Integrate your trained SSCNN' section for supported save formats.")
        return 1

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
