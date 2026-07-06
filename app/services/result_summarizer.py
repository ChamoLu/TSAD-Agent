from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


def _clean_float(value: Any, digits: int = 6) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            cleaned[key] = _clean_float(value)
        else:
            cleaned[key] = value
    return cleaned


def _top_variables(
    variable_errors: Optional[np.ndarray],
    variable_names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if variable_errors is None or variable_errors.size == 0:
        return []

    mean_errors = np.mean(variable_errors, axis=0)
    top_indices = np.argsort(mean_errors)[::-1][:top_k]
    return [
        {
            'index': int(index),
            'name': variable_names[index] if index < len(variable_names) else f'var_{index}',
            'mean_error': _clean_float(mean_errors[index]),
        }
        for index in top_indices
    ]


def merge_anomaly_windows(
    pred_labels: np.ndarray,
    scores: np.ndarray,
    variable_errors: Optional[np.ndarray],
    variable_names: List[str],
    top_k_vars: int = 5,
) -> List[Dict[str, Any]]:
    windows = []
    n_points = len(pred_labels)
    index = 0

    while index < n_points:
        if pred_labels[index] != 1:
            index += 1
            continue

        start = index
        while index + 1 < n_points and pred_labels[index + 1] == 1:
            index += 1
        end = index

        segment_scores = scores[start:end + 1]
        peak_offset = int(np.argmax(segment_scores))
        segment_errors = None
        if variable_errors is not None and len(variable_errors) >= end + 1:
            segment_errors = variable_errors[start:end + 1]

        windows.append({
            'window_index': len(windows),
            'start': int(start),
            'end': int(end),
            'duration': int(end - start + 1),
            'peak_index': int(start + peak_offset),
            'peak_score': _clean_float(segment_scores[peak_offset]),
            'mean_score': _clean_float(np.mean(segment_scores)),
            'top_variables': _top_variables(segment_errors, variable_names, top_k=top_k_vars),
        })
        index += 1

    return windows


def downsample_chart(
    scores: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: Optional[np.ndarray],
    max_points: int,
) -> List[Dict[str, Any]]:
    n_points = len(scores)
    if n_points == 0:
        return []

    step = max(1, int(math.ceil(n_points / max_points)))
    chart = []
    for start in range(0, n_points, step):
        end = min(start + step, n_points)
        segment = scores[start:end]
        local_peak = int(np.argmax(segment))
        idx = start + local_peak
        point = {
            'index': int(idx),
            'score': _clean_float(scores[idx]),
            'pred': int(pred_labels[idx]),
        }
        if true_labels is not None and len(true_labels) > idx:
            point['label'] = int(true_labels[idx])
        chart.append(point)
    return chart


def create_detection_record(
    detection: Dict[str, Any],
    config: Dict[str, Any],
    variable_names: Optional[List[str]] = None,
    max_chart_points: int = 2000,
) -> Dict[str, Any]:
    scores = np.asarray(detection['scores'], dtype=float).reshape(-1)
    pred_labels = np.asarray(detection['pred_labels'], dtype=int).reshape(-1)
    true_labels = np.asarray(detection.get('true_labels'), dtype=int).reshape(-1)
    variable_errors = detection.get('variable_errors')
    variable_errors = None if variable_errors is None else np.asarray(variable_errors, dtype=float)

    if variable_names is None:
        n_vars = variable_errors.shape[1] if variable_errors is not None and variable_errors.ndim == 2 else 0
        variable_names = [f'var_{index}' for index in range(n_vars)]

    anomaly_mask = pred_labels == 1
    anomaly_errors = variable_errors[anomaly_mask] if variable_errors is not None and np.any(anomaly_mask) else variable_errors
    windows = merge_anomaly_windows(pred_labels, scores, variable_errors, variable_names)

    summary = {
        'dataset': detection['dataset'],
        'total_points': int(len(scores)),
        'anomaly_points': int(np.sum(anomaly_mask)),
        'anomaly_point_ratio': _clean_float(np.mean(anomaly_mask) if len(scores) else 0.0),
        'anomaly_window_count': int(len(windows)),
        'threshold': _clean_float(detection['threshold']),
        'score_min': _clean_float(np.min(scores) if len(scores) else 0.0),
        'score_max': _clean_float(np.max(scores) if len(scores) else 0.0),
        'score_mean': _clean_float(np.mean(scores) if len(scores) else 0.0),
        'top_variables': _top_variables(anomaly_errors, variable_names),
        'has_true_labels': bool(true_labels is not None and len(true_labels) == len(scores)),
    }

    return {
        'id': str(uuid.uuid4()),
        'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'config': config,
        'summary': summary,
        'metrics': _clean_metrics(detection.get('metrics', {})),
        'windows': windows,
        'chart': downsample_chart(scores, pred_labels, true_labels, max_points=max_chart_points),
    }


def build_llm_context(record: Dict[str, Any], window_index: Optional[int] = None) -> str:
    windows = record.get('windows', [])
    selected_window = None
    if window_index is not None:
        for window in windows:
            if window.get('window_index') == window_index:
                selected_window = window
                break

    context = {
        'detector_config': record.get('config', {}),
        'summary': record.get('summary', {}),
        'metrics': record.get('metrics', {}),
        'selected_window': selected_window,
        'top_windows': windows[:20],
        'notes': [
            'When answering questions about this run, ground anomaly claims in the provided detector evidence.',
            'Variable contribution is detector evidence, not confirmed business root cause.',
            'If business logs, deployment events, or topology are absent, state that limitation explicitly.',
        ],
    }
    return json.dumps(context, ensure_ascii=False, indent=2)
