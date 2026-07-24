from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class FeasibleSolutionEvent:
    run_id: str
    case: str
    wall_timestamp_sec: float
    solver_incumbent_timestamp_sec: float
    cycle: int
    procedure: str
    neighborhood: str
    manifest_sha256: str
    objective_sha256: str
    structural_hash: str
    solver_objective: float
    solver_cmax: float
    verified_cmax: float
    internal_feasible: bool
    verifier_error_codes: tuple[str, ...]
    provenance: Mapping[str, Any]
    snapshot_sha256: str
    structural_projection: Mapping[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["verifier_error_codes"] = list(self.verifier_error_codes)
        row["provenance"] = dict(self.provenance)
        row["structural_projection"] = dict(self.structural_projection)
        return row


class EventLedger:
    """Append-only JSONL ledger used by the target-blind solver process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    def append(self, event: FeasibleSolutionEvent) -> None:
        if not event.internal_feasible:
            raise ValueError("only internally feasible snapshots belong in the feasible-event ledger")
        line = json.dumps(event.as_row(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        self._stream.write(line + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "EventLedger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


_FORBIDDEN_AUDIT_KEY_PARTS = (
    "target",
    "bestobjstop",
    "best_obj_stop",
    "replay",
    "solution_export",
    "historical_solution",
)


def _assert_target_blind_audit(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            compact = normalized.replace("_", "")
            if any(part in normalized or part.replace("_", "") in compact for part in _FORBIDDEN_AUDIT_KEY_PARTS):
                raise ValueError(f"search audit must remain target-blind: forbidden field {path}.{key}")
            _assert_target_blind_audit(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_target_blind_audit(item, f"{path}[{index}]")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SearchAuditLedger:
    """Append-only, target-blind process ledger for inner/outer search decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    def append(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        _assert_target_blind_audit(row)
        row["schema_version"] = 1
        line = json.dumps(_json_safe(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        self._stream.write(line + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "SearchAuditLedger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def read_event_rows(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if text:
                yield dict(json.loads(text))
