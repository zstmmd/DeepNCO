from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Gurobi.tra_engine import PaperTRAEngine
from Gurobi.tra_events import EventLedger, SearchAuditLedger
from Gurobi.tra_scheduler import RuntimeLedger
from Gurobi.tra_templates import compile_paper_tra_templates, global_config_from_policy
from Gurobi.tra_artifacts import write_verified_solution_export
from experiments.m_tra_policy import (
    PolicyError,
    assert_target_blind_payload,
    normalize_serialized_case_policy,
    runtime_budget_for_case,
)
from problemDto.createInstance import CreateOFSProblem


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _install_runtime_configs(path: Path) -> None:
    payload = _read_json(path)
    configs = payload.get("configs", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(configs, Mapping):
        raise ValueError("runtime config must contain a mapping named 'configs'")
    installed = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
    for name, cfg in configs.items():
        if isinstance(cfg, Mapping):
            installed[str(name).upper()] = dict(cfg)
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS = installed


def _load_case_policy(path: Path, case_id: str):
    payload = _read_json(path)
    assert_target_blind_payload(payload)
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise PolicyError(f"unsupported domain-policy schema: {payload.get('schema_version')}")
    cases = dict(payload.get("cases", {}) or {})
    if case_id not in cases:
        raise PolicyError(f"domain policy has no case {case_id}")
    policy = normalize_serialized_case_policy(dict(cases[case_id] or {}))
    if policy.case_id != case_id:
        raise PolicyError(f"domain policy case mismatch: requested={case_id}, payload={policy.case_id}")
    return policy


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run target-blind paper TRA-Gurobi on one M-suite case.")
    parser.add_argument("--case", required=True, choices=[f"M{index}" for index in range(1, 10)])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument(
        "--domain-policy",
        default="experiments/configs/m_tra_domain_policies_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-procedures", type=int, default=50)
    parser.add_argument("--master-domain-manifest", default="")
    parser.add_argument("--gurobi-output", action="store_true")
    return parser


def build_formal_engine(
    templates: Any,
    runtime: RuntimeLedger,
    *,
    max_procedures: int,
) -> PaperTRAEngine:
    return PaperTRAEngine(
        templates,
        runtime,
        max_procedures=int(max_procedures),
        enable_f1_live_seed_starts=False,
        enable_f1_plateau_escalation=False,
        enable_station_balance_repair=True,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    assert_target_blind_payload(vars(args))

    case_id = str(args.case).upper()
    runtime_config_path = Path(args.runtime_config)
    policy_path = Path(args.domain_policy)
    output_dir = Path(args.output_dir)
    master_domain_path = Path(args.master_domain_manifest) if str(args.master_domain_manifest or "").strip() else None
    if not runtime_config_path.is_absolute():
        runtime_config_path = ROOT_DIR / runtime_config_path
    if not policy_path.is_absolute():
        policy_path = ROOT_DIR / policy_path
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    if master_domain_path is not None and not master_domain_path.is_absolute():
        master_domain_path = ROOT_DIR / master_domain_path
    output_dir.mkdir(parents=True, exist_ok=True)

    _install_runtime_configs(runtime_config_path)
    policy = _load_case_policy(policy_path, case_id)
    budget = runtime_budget_for_case(case_id)
    problem = CreateOFSProblem.generate_problem_by_scale(case_id, seed=int(args.seed))
    cfg = global_config_from_policy(
        policy.values,
        gurobi_output=bool(args.gurobi_output),
        gurobi_seed=int(args.seed),
    )
    if master_domain_path is not None:
        cfg.master_domain_manifest = _read_json(master_domain_path)
        cfg.master_domain_strict = True

    # Canonical warm generation, both model compiles, and manifest verification are pre-timer.
    templates = compile_paper_tra_templates(
        problem,
        cfg,
        canonical_seed=int(args.seed),
        instance_name=case_id,
    )
    manifest_path = output_dir / "master_domain_v3.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(templates.manifest, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")

    runtime = RuntimeLedger(
        hard_limit_sec=float(budget.hard_limit_sec),
        inner_quota_sec=float(budget.inner_quota_sec),
        outer_quota_sec=float(budget.outer_quota_sec),
        reserve_quota_sec=float(budget.reserve_quota_sec),
        safety_buffer_sec=1.5,
        minimum_solver_slice_sec=2.0,
        objective_gap_stop=float(budget.objective_gap_stop),
    )
    engine = build_formal_engine(
        templates,
        runtime,
        max_procedures=int(args.max_procedures),
    )
    event_path = output_dir / "feasible_solution_events.jsonl"
    audit_path = output_dir / "search_audit_events.jsonl"
    with EventLedger(event_path) as ledger, SearchAuditLedger(audit_path) as audit_ledger:
        result = engine.run(
            case=case_id,
            ledger=ledger,
            audit_ledger=audit_ledger,
            run_id=str(args.run_id or "") or None,
        )

    export_error = ""
    if engine.best_verified_snapshot is None:
        solution_export = {
            "status": "NO_VERIFIED_SNAPSHOT",
            "export_complete": False,
        }
    else:
        try:
            solution_export = write_verified_solution_export(
                output_dir / "paper_tra_solution_export",
                verified=engine.best_verified_snapshot,
                case_id=case_id,
                seed=int(args.seed),
                run_id=result.run_id,
                manifest_sha256=result.manifest_sha256,
            )
            solution_export = {
                "status": "EXPORTED",
                **solution_export,
            }
        except Exception as exc:
            export_error = str(exc)
            solution_export = {
                "status": "EXPORT_FAILED",
                "export_complete": False,
                "error": export_error,
            }

    summary = {
        "schema_version": 1,
        "algorithm": "paper-tra-gurobi",
        "case": case_id,
        "seed": int(args.seed),
        "domain_policy_sha256": policy.policy_sha256,
        "runtime_policy": asdict(budget),
        "manifest_path": str(manifest_path),
        "event_ledger_path": str(event_path),
        "search_audit_ledger_path": str(audit_path),
        "solution_export": solution_export,
        "result": asdict(result),
    }
    assert_target_blind_payload(
        {
            "case": case_id,
            "seed": int(args.seed),
            "domain_policy_sha256": policy.policy_sha256,
            "runtime_sec": result.runtime_sec,
        }
    )
    summary_path = output_dir / "tra_gurobi_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(_json_safe(summary), ensure_ascii=True, sort_keys=True))
    if export_error:
        raise RuntimeError(f"verified solution export failed: {export_error}")


if __name__ == "__main__":
    main()
