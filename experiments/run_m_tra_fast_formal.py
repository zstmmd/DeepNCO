from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Gurobi.tra_events import EventLedger, SearchAuditLedger
from Gurobi.tra_fast_engine import PaperTRAFastEngine
from Gurobi.tra_fast_search import PaperFastNeighborhoodTemplate
from Gurobi.tra_model_state import PersistentCompiledTemplate
from Gurobi.tra_scheduler import RuntimeLedger
from Gurobi.tra_templates import compile_paper_tra_templates, global_config_from_policy
from experiments.m_tra_fast_policy import load_fast_runtime_budget
from experiments.m_tra_policy import assert_target_blind_payload
from experiments.run_m_tra_gurobi_formal import (
    _install_runtime_configs,
    _load_case_policy,
)
from problemDto.createInstance import CreateOFSProblem


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run target-blind paper TRA-Fast on one M-suite case."
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=[f"M{index}" for index in range(1, 10)],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument(
        "--domain-policy",
        default="experiments/configs/m_tra_domain_policies_v1.json",
    )
    parser.add_argument(
        "--fast-runtime-policy",
        default="experiments/configs/m_tra_fast_runtime_budgets_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-procedures", type=int, default=50)
    parser.add_argument("--gurobi-output", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_target_blind_payload(vars(args))
    case_id = str(args.case).upper()

    def absolute(raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else ROOT_DIR / path

    runtime_config_path = absolute(args.runtime_config)
    policy_path = absolute(args.domain_policy)
    fast_policy_path = absolute(args.fast_runtime_policy)
    output_dir = absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _install_runtime_configs(runtime_config_path)
    policy = _load_case_policy(policy_path, case_id)
    budget = load_fast_runtime_budget(fast_policy_path, case_id)
    problem = CreateOFSProblem.generate_problem_by_scale(case_id, seed=int(args.seed))
    cfg = global_config_from_policy(
        policy.values,
        gurobi_output=bool(args.gurobi_output),
        gurobi_seed=int(args.seed),
    )
    templates = compile_paper_tra_templates(
        problem,
        cfg,
        canonical_seed=int(args.seed),
        instance_name=case_id,
    )
    fast = PaperFastNeighborhoodTemplate(
        PersistentCompiledTemplate(
            templates.full_compiled,
            solver=templates.outer.template.solver,
        ),
        verifier=templates.outer.verifier,
    )
    manifest_path = output_dir / "master_domain_v3.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(
            templates.manifest,
            stream,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    runtime = RuntimeLedger(
        hard_limit_sec=budget.hard_limit_sec,
        inner_quota_sec=0.0,
        outer_quota_sec=budget.regular_quota_sec,
        reserve_quota_sec=budget.reserve_quota_sec,
        safety_buffer_sec=0.5,
    )
    engine = PaperTRAFastEngine(
        templates,
        fast,
        runtime,
        max_procedures=int(args.max_procedures),
    )
    event_path = output_dir / "feasible_solution_events.jsonl"
    audit_path = output_dir / "search_audit_events.jsonl"
    with EventLedger(event_path) as ledger, SearchAuditLedger(audit_path) as audit:
        result = engine.run(
            case=case_id,
            ledger=ledger,
            audit_ledger=audit,
            run_id=str(args.run_id or "") or None,
        )

    summary = {
        "schema_version": 1,
        "algorithm": "paper-tra-fast",
        "case": case_id,
        "seed": int(args.seed),
        "domain_policy_sha256": policy.policy_sha256,
        "runtime_policy": asdict(budget),
        "manifest_path": str(manifest_path),
        "event_ledger_path": str(event_path),
        "search_audit_ledger_path": str(audit_path),
        "result": asdict(result),
    }
    summary_path = output_dir / "tra_fast_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_safe(summary),
            stream,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(json.dumps(_json_safe(summary), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
