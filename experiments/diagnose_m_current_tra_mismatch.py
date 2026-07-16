from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parents[1]


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def _export_status(export_dir: Path) -> Dict[str, Any]:
    verification_txt = _read_text(export_dir / "tra_makespan_verification.txt")
    audit_txt = _read_text(export_dir / "best_solution_audit.txt")
    verification_json = _read_json(export_dir / "tra_makespan_verification.json", {})
    audit_json = _read_json(export_dir / "best_solution_audit.json", {})
    return {
        "export_dir": os.path.relpath(export_dir, ROOT_DIR) if export_dir.exists() else str(export_dir),
        "export_exists": export_dir.exists(),
        "verification_txt_exists": bool(verification_txt),
        "audit_txt_exists": bool(audit_txt),
        "verification_pass": "status=PASS" in verification_txt or str(verification_json.get("status", "")).upper() == "PASS",
        "coverage_ok": "coverage_ok=True" in verification_txt or "coverage_ok=True" in audit_txt or bool(audit_json.get("coverage_ok", False)),
        "makespan_consistent": "makespan_consistent=True" in audit_txt or bool(audit_json.get("makespan_consistent", False)),
        "has_unreasonable_solution": "has_unreasonable_solution=True" in audit_txt or bool(audit_json.get("has_unreasonable_solution", False)),
    }


def _alias_status(runtime_config_json: Path, case: str) -> Dict[str, Any]:
    payload = _read_json(runtime_config_json, {})
    configs = payload.get("configs", payload) if isinstance(payload, dict) else {}
    source_case = case.replace("GUROBI-", "", 1) if case.startswith("GUROBI-") else case
    alias_case = case if case.startswith("GUROBI-") else f"GUROBI-{case}"
    source_cfg = configs.get(source_case)
    alias_cfg = configs.get(alias_case)
    return {
        "runtime_config_json": os.path.relpath(runtime_config_json, ROOT_DIR) if runtime_config_json.exists() else str(runtime_config_json),
        "source_case": source_case,
        "alias_case": alias_case,
        "source_exists": isinstance(source_cfg, dict),
        "alias_exists": isinstance(alias_cfg, dict),
        "source_alias_equal": bool(isinstance(source_cfg, dict) and isinstance(alias_cfg, dict) and source_cfg == alias_cfg),
    }


def diagnose(
    *,
    case: str,
    algorithm: str,
    gurobi_summary: Path,
    gurobi_export_dir: Path,
    tra_export_dir: Path,
    runtime_config_json: Path,
) -> Dict[str, Any]:
    case = str(case).upper()
    if not case.startswith("GUROBI-"):
        case = f"GUROBI-{case}"
    gurobi = _read_json(gurobi_summary, {})
    gurobi_diag = dict(gurobi.get("diagnostics", {}) or {})
    alias = _alias_status(runtime_config_json, case)
    gurobi_export = _export_status(gurobi_export_dir)
    tra_export = _export_status(tra_export_dir)
    findings = []
    if not alias["source_alias_equal"]:
        findings.append("baseline_alias_mismatch")
    if not gurobi_export["verification_pass"]:
        findings.append("gurobi_verification_missing_or_failed")
    if not tra_export["verification_pass"]:
        findings.append("tra_verification_missing_or_failed")
    if not tra_export["makespan_consistent"]:
        findings.append("tra_makespan_inconsistent")
    if bool(gurobi_diag.get("time_verify_mismatch", False)):
        findings.append("gurobi_time_verify_mismatch")
    if int(gurobi_diag.get("warm_start_missing_arc_count", 0) or 0) > 0:
        findings.append("gurobi_warm_start_missing_arc")
    if not findings:
        diagnosis_status = "no_mismatch_found_but_lower_cmax_unacceptable"
    elif any(item in findings for item in ("baseline_alias_mismatch", "tra_verification_missing_or_failed", "tra_makespan_inconsistent")):
        diagnosis_status = "constraint_mismatch_suspected"
    else:
        diagnosis_status = "evidence_insufficient"
    return {
        "case": case,
        "algorithm": str(algorithm),
        "diagnosis_status": diagnosis_status,
        "findings": findings,
        "alias_status": alias,
        "gurobi_summary": os.path.relpath(gurobi_summary, ROOT_DIR) if gurobi_summary.exists() else str(gurobi_summary),
        "gurobi_cmax": gurobi.get("global_makespan", gurobi_diag.get("model_cmax")),
        "gurobi_status": gurobi.get("status"),
        "gurobi_export_status": gurobi_export,
        "tra_export_status": tra_export,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose current M-suite TRA/Gurobi mismatch.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--algorithm", required=True, choices=("tra_gurobi", "tra_fast"))
    parser.add_argument("--gurobi-summary", required=True)
    parser.add_argument("--gurobi-export-dir", required=True)
    parser.add_argument("--tra-export-dir", required=True)
    parser.add_argument("--runtime-config-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = diagnose(
        case=args.case,
        algorithm=args.algorithm,
        gurobi_summary=_resolve(args.gurobi_summary),
        gurobi_export_dir=_resolve(args.gurobi_export_dir),
        tra_export_dir=_resolve(args.tra_export_dir),
        runtime_config_json=_resolve(args.runtime_config_json),
    )
    out = _resolve(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
