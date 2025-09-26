"""Report generation utilities for optimisation runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _objective_iterator(objectives: Iterable[object]) -> Iterable[Tuple[str, float]]:
    for obj in objectives:
        if isinstance(obj, str):
            yield obj, 1.0
        elif isinstance(obj, dict):
            name = obj.get("name") or obj.get("metric")
            if not name:
                continue
            weight = float(obj.get("weight", 1.0))
            yield str(name), weight


def _flatten_results(results: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aggregated_rows: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []

    for record in results:
        base_row: Dict[str, object] = {
            "trial": record.get("trial"),
            "score": record.get("score"),
            "valid": record.get("valid", True),
        }
        base_row.update(record.get("params", {}))
        for key, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float, bool)):
                base_row[key] = value
        aggregated_rows.append(base_row)

        for dataset in record.get("datasets", []):
            ds_row: Dict[str, object] = {
                "trial": record.get("trial"),
                "score": record.get("score"),
                "valid": dataset.get("metrics", {}).get("Valid", True),
                "dataset": dataset.get("name"),
            }
            ds_row.update(dataset.get("meta", {}))
            ds_row.update(record.get("params", {}))
            for key, value in dataset.get("metrics", {}).items():
                if isinstance(value, (int, float, bool)):
                    ds_row[key] = value
            dataset_rows.append(ds_row)

    return pd.DataFrame(aggregated_rows), pd.DataFrame(dataset_rows)


def _annotate_objectives(df: pd.DataFrame, objectives: Iterable[object]) -> pd.DataFrame:
    if df.empty:
        return df

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for name, weight in _objective_iterator(objectives):
        if name not in df.columns:
            continue
        series = df[name].astype(float)
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (series - series.mean()) / std
        df[f"{name}_z"] = z
        composite += weight * z
        total_weight += abs(weight)

    if total_weight:
        df["CompositeScore"] = composite / total_weight
    else:
        df["CompositeScore"] = composite
    return df


def export_results(results: List[Dict[str, object]], objectives: Iterable[object], output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dir(output_dir)
    agg_df, dataset_df = _flatten_results(results)
    agg_df = _annotate_objectives(agg_df, objectives)
    agg_df.to_csv(output_dir / "results.csv", index=False)
    if not dataset_df.empty:
        dataset_df.to_csv(output_dir / "results_datasets.csv", index=False)
    return agg_df, dataset_df


def export_best(best: Dict[str, object], wf_summary: Dict[str, object], output_dir: Path) -> None:
    segments_payload = []
    for seg in wf_summary.get("segments", []):
        segments_payload.append(
            {
                "train": [seg.train_start.isoformat(), seg.train_end.isoformat()],
                "test": [seg.test_start.isoformat(), seg.test_end.isoformat()],
                "train_metrics": seg.train_metrics,
                "test_metrics": seg.test_metrics,
            }
        )

    payload = {
        "params": best.get("params"),
        "metrics": best.get("metrics"),
        "score": best.get("score"),
        "datasets": best.get("datasets", []),
        "walk_forward": {
            "oos_mean": wf_summary.get("oos_mean"),
            "oos_median": wf_summary.get("oos_median"),
            "count": wf_summary.get("count"),
            "segments": segments_payload,
            "candidates": wf_summary.get("candidates", []),
        },
    }
    (output_dir / "best.json").write_text(json.dumps(payload, indent=2))


def export_heatmap(metrics_df: pd.DataFrame, params: List[str], metric: str, output_dir: Path) -> None:
    if len(params) < 2 or metrics_df.empty or metric not in metrics_df.columns:
        return
    x_param, y_param = params[:2]
    if x_param not in metrics_df or y_param not in metrics_df:
        return
    pivot = metrics_df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc="mean")
    if pivot.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=False, cmap="viridis")
    plt.title(f"{metric} heatmap ({y_param} vs {x_param})")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap.png")
    plt.close()


def generate_reports(
    results: List[Dict[str, object]],
    best: Dict[str, object],
    wf_summary: Dict[str, object],
    objectives: Iterable[object],
    output_dir: Path,
) -> None:
    agg_df, _ = export_results(results, objectives, output_dir)
    export_best(best, wf_summary, output_dir)

    params = list(best.get("params", {}).keys())
    metric_name = next((name for name, _ in _objective_iterator(objectives)), "NetProfit")
    export_heatmap(agg_df, params, metric_name, output_dir)
