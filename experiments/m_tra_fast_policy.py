from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.m_tra_policy import PolicyError, assert_target_blind_payload


@dataclass(frozen=True)
class FastRuntimeBudget:
    case_id: str
    source_tra_gurobi_runtime_sec: float
    hard_limit_sec: float
    regular_quota_sec: float
    reserve_quota_sec: float
    policy_sha256: str


def _sha256(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_fast_runtime_budget(path: Path, case_id: str) -> FastRuntimeBudget:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert_target_blind_payload(payload)
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise PolicyError("unsupported TRA-Fast runtime-policy schema")
    case = dict(dict(payload.get("cases", {}) or {}).get(str(case_id), {}) or {})
    if not case:
        raise PolicyError(f"TRA-Fast runtime policy has no case {case_id}")
    source = float(case["source_tra_gurobi_runtime_sec"])
    hard = float(case.get("hard_limit_sec", 0.8 * source))
    if source <= 0.0 or hard <= 0.0 or hard > 0.8 * source + 1e-9:
        raise PolicyError(f"invalid TRA-Fast runtime budget for {case_id}")
    canonical = {
        "case_id": str(case_id),
        "source_tra_gurobi_runtime_sec": source,
        "hard_limit_sec": hard,
    }
    return FastRuntimeBudget(
        case_id=str(case_id),
        source_tra_gurobi_runtime_sec=source,
        hard_limit_sec=hard,
        regular_quota_sec=0.85 * hard,
        reserve_quota_sec=0.15 * hard,
        policy_sha256=_sha256(canonical),
    )
