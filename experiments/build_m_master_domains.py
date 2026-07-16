from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from Gurobi.master_domain import build_master_domain_manifest
from experiments.m_current_tra_baselines import CurrentMCase, load_current_m_cases
from problemDto.createInstance import CreateOFSProblem


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return dict(json.load(stream) or {})


def _install_case_runtime_config(case: CurrentMCase) -> None:
    payload = _read_json(case.config_abs_path)
    configs = dict(payload.get("configs", payload) or {})
    case_config = dict(configs.get(case.case_id, {}) or {})
    if not case_config:
        raise KeyError(f"missing runtime config for {case.case_id} in {case.config_abs_path}")
    installed = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
    installed[case.case_id] = case_config
    installed[case.algorithm_case] = case_config
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS = installed


def canonical_global_config(summary: Dict[str, Any]) -> GlobalXYZUConfig:
    raw = dict(summary.get("config", {}) or {})
    allowed = {item.name for item in fields(GlobalXYZUConfig)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    kwargs.update(
        {
            "write_lp": False,
            "gurobi_output": False,
            "gurobi_cutoff": None,
            "gurobi_best_obj_stop": None,
            "master_domain_manifest": None,
            "master_domain_strict": False,
        }
    )
    return GlobalXYZUConfig(**kwargs)


def build_case_master_domain(case: CurrentMCase, *, output_dir: Path, seed: int = 42) -> Path:
    _install_case_runtime_config(case)
    summary = _read_json(case.gurobi_summary_abs_path)
    problem = CreateOFSProblem.generate_problem_by_scale(case.algorithm_case, seed=int(seed))
    cfg = canonical_global_config(summary)
    compiled = GlobalXYZUSolver().compile_model(problem, cfg)
    try:
        manifest = build_master_domain_manifest(
            compiled,
            canonical_seed=int(seed),
            instance_name=case.algorithm_case,
        )
    finally:
        dispose = getattr(compiled.model, "dispose", None)
        if callable(dispose):
            dispose()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.algorithm_case}_master_domain.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=True, indent=2)
    return output_path


def build_master_domains(
    cases: Iterable[CurrentMCase],
    *,
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for case in cases:
        path = build_case_master_domain(case, output_dir=output_dir, seed=seed)
        paths[case.algorithm_case] = str(path)
    map_path = output_dir / "master_domain_map.json"
    with map_path.open("w", encoding="utf-8") as stream:
        json.dump({"manifests": paths}, stream, ensure_ascii=True, indent=2)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-blind shared M-suite master domains.")
    parser.add_argument("--cases", nargs="+", default=[f"M{i}" for i in range(1, 10)])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="result/m_current_master_domains")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {str(value).upper().replace("GUROBI-", "") for value in args.cases}
    cases = [case for case in load_current_m_cases() if case.case_id in selected]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    build_master_domains(cases, output_dir=output_dir, seed=int(args.seed))


if __name__ == "__main__":
    main()
