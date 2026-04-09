from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class CandidatePrediction:
    prior_score: float
    win_prob: float
    win_prob_std: float
    residual_hat: float
    residual_std: float
    uncertainty: float
    predicted_proxy_z: float


@dataclass
class F1EvalResult:
    proxy_z: float
    station_cmax: float
    route_tail: float
    mapping_coverage: float
    replayed_route: bool
    used_full_replay: bool
    replayed_trip_count: int
    arrival_shift_total: float = 0.0
    wait_overflow_total: float = 0.0
    route_tail_delta: float = 0.0
    used_sp2: bool = False
    used_sp3: bool = False
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingSample:
    f0_features: Dict[str, float]
    f1_features: Dict[str, float]
    label: int
    residual: float
    sample_weight: float
    candidate_signature: str
    predicted_win_prob: float
    predicted_proxy_z: float
    actual_proxy_z: float


class OnlineFeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return bool(self._fitted)

    def partial_fit(self, rows: Sequence[Sequence[float]]) -> None:
        if not rows:
            return
        array = np.asarray(list(rows), dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        self.scaler.partial_fit(array)
        self._fitted = True

    def transform(self, rows: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(list(rows), dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if not self._fitted:
            return array
        return self.scaler.transform(array)


class OnlineBinaryRankEnsemble:
    def __init__(self, size: int = 3, alpha: float = 1e-4, random_seed: int = 42):
        self.models: List[SGDClassifier] = [
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=float(alpha),
                learning_rate="optimal",
                random_state=int(random_seed + idx * 17),
            )
            for idx in range(max(1, int(size)))
        ]
        self._initialized = False

    @property
    def fitted(self) -> bool:
        return bool(self._initialized)

    def partial_fit(self, x_rows: np.ndarray, y_rows: Sequence[int], sample_weight: Optional[Sequence[float]] = None) -> None:
        if x_rows is None or len(x_rows) == 0:
            return
        y_arr = np.asarray(list(y_rows), dtype=int)
        weight_arr = None if sample_weight is None else np.asarray(list(sample_weight), dtype=float)
        for model in self.models:
            if not self._initialized:
                model.partial_fit(x_rows, y_arr, classes=np.asarray([0, 1], dtype=int), sample_weight=weight_arr)
            else:
                model.partial_fit(x_rows, y_arr, sample_weight=weight_arr)
        self._initialized = True

    def predict_mean_std(self, x_rows: np.ndarray) -> np.ndarray:
        if x_rows is None or len(x_rows) == 0:
            return np.zeros((0, 2), dtype=float)
        if not self._initialized:
            return np.column_stack([
                np.full(len(x_rows), 0.5, dtype=float),
                np.full(len(x_rows), 0.5, dtype=float),
            ])
        preds = []
        for model in self.models:
            proba = model.predict_proba(x_rows)
            preds.append(np.asarray(proba[:, 1], dtype=float))
        pred_arr = np.vstack(preds)
        return np.column_stack([
            np.mean(pred_arr, axis=0),
            np.std(pred_arr, axis=0),
        ])


class OnlineResidualEnsemble:
    def __init__(self, size: int = 3, alpha: float = 1e-4, random_seed: int = 42):
        self.models: List[SGDRegressor] = [
            SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=float(alpha),
                learning_rate="invscaling",
                eta0=0.01,
                power_t=0.25,
                random_state=int(random_seed + idx * 29),
            )
            for idx in range(max(1, int(size)))
        ]
        self._initialized = False

    @property
    def fitted(self) -> bool:
        return bool(self._initialized)

    def partial_fit(self, x_rows: np.ndarray, y_rows: Sequence[float], sample_weight: Optional[Sequence[float]] = None) -> None:
        if x_rows is None or len(x_rows) == 0:
            return
        y_arr = np.asarray(list(y_rows), dtype=float)
        weight_arr = None if sample_weight is None else np.asarray(list(sample_weight), dtype=float)
        for model in self.models:
            model.partial_fit(x_rows, y_arr, sample_weight=weight_arr)
        self._initialized = True

    def predict_mean_std(self, x_rows: np.ndarray) -> np.ndarray:
        if x_rows is None or len(x_rows) == 0:
            return np.zeros((0, 2), dtype=float)
        if not self._initialized:
            return np.column_stack([
                np.zeros(len(x_rows), dtype=float),
                np.zeros(len(x_rows), dtype=float),
            ])
        preds = []
        for model in self.models:
            preds.append(np.asarray(model.predict(x_rows), dtype=float))
        pred_arr = np.vstack(preds)
        return np.column_stack([
            np.mean(pred_arr, axis=0),
            np.std(pred_arr, axis=0),
        ])


@dataclass
class SurrogateLayerState:
    layer: str
    feature_names_f0: List[str]
    feature_names_f1: List[str]
    scaler_f0: OnlineFeatureScaler
    scaler_f1: OnlineFeatureScaler
    rank_models: OnlineBinaryRankEnsemble
    residual_models: OnlineResidualEnsemble
    warmup_count: int = 0
    training_buffer: Deque[TrainingSample] = field(default_factory=lambda: deque(maxlen=512))
    f1_cache: Dict[str, F1EvalResult] = field(default_factory=dict)
    uncertainty_probe_counter: int = 0
    anchor_version: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0
    interval_hit_count: int = 0
    rank_hit_top1_count: int = 0
    rank_top1_total: int = 0

    def make_f0_vector(self, features: Dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names_f0]

    def make_f1_vector(self, features: Dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names_f1]

