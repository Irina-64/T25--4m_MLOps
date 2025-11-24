import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Register model if metric above threshold")
    p.add_argument("--model-path", required=True, help="Path to trained model (dir)")
    p.add_argument("--registry-path", required=True, help="Path to production registry (dir)")
    p.add_argument("--report-path", required=True, help="Path to evaluation report (JSON)")
    p.add_argument("--metric", default="bleu", help="Metric name in the report")
    p.add_argument("--threshold", type=float, default=0.2, help="Threshold to accept model")
    return p.parse_args()

def main():
    args = parse_args()

    report_file = Path(args.report_path)
    if not report_file.exists():
        raise FileNotFoundError(f"Report file not found: {report_file}")

    with report_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metric_value = float(data.get(args.metric, 0.0))

    model_src = Path(args.model_path)
    model_dst = Path(args.registry_path)
    model_dst.parent.mkdir(parents=True, exist_ok=True)

    if metric_value >= args.threshold:
        if model_dst.exists():
            shutil.rmtree(model_dst)
        shutil.copytree(model_src, model_dst)

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "model_registered.txt").write_text(
            f"REGISTERED: {args.metric}={metric_value:.4f} >= {args.threshold} | source={model_src}\n",
            encoding="utf-8",
        )
        print(f"[REGISTER] Model accepted: {args.metric} {metric_value:.4f} >= {args.threshold}")
    else:
        raise SystemExit(
            f"[REJECT] {args.metric} {metric_value:.4f} below threshold {args.threshold}"
        )


if __name__ == "__main__":
    main()