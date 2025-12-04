"""
Simple drift detection script.

It compares feature distributions between the training reference set and the
most recent production requests using PSI/KS statistics and checks the current
model quality (BLEU as a ROC-AUC surrogate for this text task). If any signal
crosses thresholds, it can optionally trigger retraining.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from evaluate import load as load_metric
from scipy.stats import ks_2samp
from transformers import T5ForConditionalGeneration, T5Tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "raw" / "data.csv"
DEFAULT_PROD_PATH = BASE_DIR / "data" / "production" / "requests.csv"
DEFAULT_CONTROL_PATH = BASE_DIR / "data" / "processed" / "test.csv"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "detox_model"
DEFAULT_REPORT = BASE_DIR / "reports" / "drift_report.json"
DEFAULT_BASELINE_EVAL = BASE_DIR / "reports" / "eval.json"


@dataclass
class DriftResult:
    psi: Dict[str, float]
    ks_pvalue: Dict[str, float]
    metric_baseline: Optional[float]
    metric_current: Optional[float]
    metric_drop: Optional[float]
    drift_reasons: List[str]
    drift_detected: bool
    triggered_retrain: bool


def load_dataframe(path: Path, required_cols: Tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file at {path} is missing")
    df = pd.read_csv(path)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derive simple numeric features for drift checks."""
    feats: Dict[str, pd.Series] = {}
    if "input_text" in df:
        feats["input_length"] = df["input_text"].astype(str).str.len()
        feats["input_word_count"] = df["input_text"].astype(str).str.split().apply(len)
    if "target_text" in df:
        feats["target_length"] = df["target_text"].astype(str).str.len()
        feats["target_word_count"] = df["target_text"].astype(str).str.split().apply(len)
    return pd.DataFrame(feats)


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for one feature."""
    expected_perc, bin_edges = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bin_edges)

    expected_perc = expected_perc / np.maximum(expected_perc.sum(), 1)
    actual_perc = actual_perc / np.maximum(actual_perc.sum(), 1)

    epsilon = 1e-6
    psi_values = (actual_perc - expected_perc) * np.log(
        (actual_perc + epsilon) / (expected_perc + epsilon)
    )
    return float(np.sum(psi_values))


def compare_distributions(ref: pd.DataFrame, prod: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    psi_scores: Dict[str, float] = {}
    ks_pvalues: Dict[str, float] = {}
    common_cols = set(ref.columns) & set(prod.columns)
    for col in sorted(common_cols):
        psi_scores[col] = compute_psi(ref[col], prod[col])
        ks_pvalues[col] = float(ks_2samp(ref[col], prod[col]).pvalue)
    return psi_scores, ks_pvalues


def load_baseline_metric(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("bleu", "roc_auc", "roc_auc_score"):
        if key in data:
            return float(data[key])
    return None


def evaluate_model_bleu(model_dir: Path, control_df: pd.DataFrame, max_samples: int = 200) -> float:
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    bleu = load_metric("bleu")
    preds: List[str] = []
    refs: List[List[str]] = []

    subset = control_df.head(max_samples)
    for _, row in subset.iterrows():
        inp = str(row["input_text"])
        ref = str(row["target_text"])
        inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append([ref])

    result = bleu.compute(predictions=preds, references=refs)
    return float(result["bleu"])


def decide_metric_drop(baseline: Optional[float], current: Optional[float]) -> Optional[float]:
    if baseline is None or current is None or baseline == 0:
        return None
    return (baseline - current) / baseline


def run_retrain(command: str) -> bool:
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except Exception as exc:
        print(f"[drift_check] Retrain command failed: {exc}")
        return False


def detect_drift(
    train_path: Path,
    prod_path: Path,
    control_path: Path,
    model_dir: Path,
    report_path: Path,
    psi_threshold: float,
    ks_pvalue_threshold: float,
    metric_drop_threshold: float,
    retrain_cmd: Optional[str],
    max_eval_samples: int,
) -> DriftResult:
    train_df = load_dataframe(train_path, ("input_text", "target_text"))
    prod_df = load_dataframe(prod_path, ("input_text",))

    ref_feats = build_feature_frame(train_df)
    prod_feats = build_feature_frame(prod_df)

    psi_scores, ks_pvalues = compare_distributions(ref_feats, prod_feats)

    baseline_metric = load_baseline_metric(DEFAULT_BASELINE_EVAL)

    current_metric = None
    metric_drop = None
    if control_path.exists() and model_dir.exists():
        control_df = load_dataframe(control_path, ("input_text", "target_text"))
        current_metric = evaluate_model_bleu(model_dir, control_df, max_samples=max_eval_samples)
        metric_drop = decide_metric_drop(baseline_metric, current_metric)

    drift_reasons: List[str] = []
    for name, score in psi_scores.items():
        if score > psi_threshold:
            drift_reasons.append(f"PSI({name})={score:.3f} > {psi_threshold}")
    for name, pvalue in ks_pvalues.items():
        if pvalue < ks_pvalue_threshold:
            drift_reasons.append(f"KS({name}) p={pvalue:.4f} < {ks_pvalue_threshold}")
    if metric_drop is not None and metric_drop > metric_drop_threshold:
        drift_reasons.append(f"Metric drop {metric_drop:.2%} > {metric_drop_threshold:.2%}")

    drift_detected = len(drift_reasons) > 0

    triggered = False
    if drift_detected and retrain_cmd:
        triggered = run_retrain(retrain_cmd)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "psi": psi_scores,
                "ks_pvalue": ks_pvalues,
                "metric_baseline": baseline_metric,
                "metric_current": current_metric,
                "metric_drop": metric_drop,
                "drift_reasons": drift_reasons,
                "drift_detected": drift_detected,
                "triggered_retrain": triggered,
                "thresholds": {
                    "psi": psi_threshold,
                    "ks_pvalue": ks_pvalue_threshold,
                    "metric_drop": metric_drop_threshold,
                },
            },
            f,
            indent=2,
        )

    status = "DRIFT" if drift_detected else "OK"
    print(f"[drift_check] Status: {status}")
    if drift_reasons:
        print("[drift_check] Reasons:")
        for reason in drift_reasons:
            print(f"  - {reason}")
    print(f"[drift_check] Report written to {report_path}")

    return DriftResult(
        psi=psi_scores,
        ks_pvalue=ks_pvalues,
        metric_baseline=baseline_metric,
        metric_current=current_metric,
        metric_drop=metric_drop,
        drift_reasons=drift_reasons,
        drift_detected=drift_detected,
        triggered_retrain=triggered,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drift detection for detox model.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--prod-path", type=Path, default=DEFAULT_PROD_PATH)
    parser.add_argument("--control-path", type=Path, default=DEFAULT_CONTROL_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--psi-threshold", type=float, default=float(os.getenv("PSI_THRESHOLD", 0.2)))
    parser.add_argument("--ks-pvalue-threshold", type=float, default=float(os.getenv("KS_PVALUE_THRESHOLD", 0.05)))
    parser.add_argument(
        "--metric-drop-threshold", type=float, default=float(os.getenv("METRIC_DROP_THRESHOLD", 0.1))
    )
    parser.add_argument(
        "--retrain-cmd",
        type=str,
        default=None,
        help="Shell command to execute when drift is detected (e.g. `python src/train.py`).",
    )
    parser.add_argument("--max-eval-samples", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    detect_drift(
        train_path=args.train_path,
        prod_path=args.prod_path,
        control_path=args.control_path,
        model_dir=args.model_dir,
        report_path=args.report_path,
        psi_threshold=args.psi_threshold,
        ks_pvalue_threshold=args.ks_pvalue_threshold,
        metric_drop_threshold=args.metric_drop_threshold,
        retrain_cmd=args.retrain_cmd,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
