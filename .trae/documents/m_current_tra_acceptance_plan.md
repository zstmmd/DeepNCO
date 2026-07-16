# M1-M9 当前基线 TRA-Gurobi / TRA-Fast 验收计划

## Summary

目标是在当前 M1-M9 Gurobi 验收基线下重新运行 TRA-Gurobi 与 TRA-Fast，并按论文安全口径验收：

- TRA-Gurobi Cmax 必须与当前 Gurobi Cmax 一致。
- TRA-Gurobi runtime 必须 `<= 0.8 * Gurobi runtime`。
- TRA-Fast Cmax 必须与当前 Gurobi Cmax 一致。
- TRA-Fast runtime 必须 `<= 0.8 * TRA-Gurobi runtime`。
- 任一 TRA 解低于 Gurobi Cmax，立即停止并诊断约束/实例/候选集是否不一致。

当前脚本中的旧 M-suite 常量仍是 `GUROBI-M1=489`、`GUROBI-M9=731` 一类旧口径，不能直接用于本轮实验。因此本轮采用“当前 baseline artifacts + 当前 runtime alias + 分层验收 runner”的方式执行。

## Current State Analysis

当前 Gurobi 基线来自 2026-07-14 已整理的 M1-M9 结果：

| Case | Cmax | Runtime(s) | Gap | Status |
| --- | ---: | ---: | ---: | --- |
| M1 | 582.0 | 360.180 | 0.005407 | TIME_LIMIT |
| M2 | 805.0 | 384.041 | 0.009069 | OPTIMAL |
| M3 | 830.0 | 663.531 | 0.009855 | OPTIMAL |
| M4 | 1098.0 | 567.059 | 0.009321 | OPTIMAL |
| M5 | 863.000001 | 663.667 | 0.009499 | OPTIMAL |
| M6 | 1064.0 | 1039.615 | 0.009207 | OPTIMAL |
| M7 | 1538.0 | 1314.206 | 0.009405 | OPTIMAL |
| M8 | 1411.0 | 1837.859 | 0.008794 | OPTIMAL |
| M9 | 2110.0 | 2608.732 | 0.009777 | OPTIMAL |

参考论文《How to Deploy Robotic Mobile Fulfillment Systems》的核心对照是三阶段快速方法与集成决策方法。本轮实验的论文创新点应表述为：在同一 Global XYZU 约束口径下，构造 `Gurobi baseline -> TRA-Gurobi exact-aligned refinement -> TRA-Fast calibrated acceleration` 的分层求解链，并用 lower-than-Gurobi fail-fast 守门避免把口径不一致误写成算法优势。

## Proposed Changes

### 1. Baseline 输入源

新增 `experiments/m_current_tra_baselines.py`：

- 从当前 9 个 `gurobi_summary.json` 读取 Cmax、runtime、gap、bound、变量数、约束数、验证字段。
- 生成 `current_m_gurobi_baseline.json/csv`。
- 生成 `current_m_runtime_aliases.json`，同时包含 `M1` 和 `GUROBI-M1` 等 alias，保证 TRA 脚本不会落回旧内置实例。

### 2. 分层验收 Runner

新增 `experiments/run_m_current_tra_acceptance.py`：

- 逐 case 运行 TRA-Gurobi。
- TRA-Gurobi 通过后，再以其 runtime 写入 TRA-Fast baseline 的 `current_tra_sec`。
- 逐 case 运行 TRA-Fast。
- 每一步立即检查 Cmax equality、lower-than-Gurobi、20% speed gate。
- 默认 `--stop-on-first-fail`。

### 3. 一致性诊断

新增 `experiments/diagnose_m_current_tra_mismatch.py`：

- 检查 runtime alias 是否一致。
- 检查 Gurobi/TRA export、`tra_makespan_verification`、`best_solution_audit`。
- lower-than-Gurobi 时输出机器可读诊断 JSON。

### 4. 测试

新增 `tests/test_m_current_tra_acceptance.py`：

- baseline row 生成。
- runtime alias 生成。
- Cmax lower-than-Gurobi fail-fast。
- Cmax 浮点噪声容忍。
- 两层 20% speed gate。
- missing/nonfinite TRA summary 分类。

## Verification Steps

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m py_compile \
  experiments/m_current_tra_baselines.py \
  experiments/run_m_current_tra_acceptance.py \
  experiments/diagnose_m_current_tra_mismatch.py

python3 -m pytest tests/test_m_current_tra_acceptance.py -q
```

正式验收命令：

```bash
PYTHONUNBUFFERED=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 experiments/run_m_current_tra_acceptance.py \
  --cases M1 M2 M3 M4 M5 M6 M7 M8 M9 \
  --seed 42 \
  --min-tra-gurobi-speedup 0.20 \
  --min-tra-fast-speedup 0.20 \
  --cmax-abs-tol 1e-5 \
  --stop-on-lower-cmax \
  --stop-on-first-fail \
  --output-root result/m_current_tra_acceptance_20260714
```

## Execution Result

本轮已执行正式命令，并按 `--stop-on-first-fail` 停在 `GUROBI-M1`：

- Gurobi M1 baseline: Cmax `582.0`, runtime `360.18s`。
- TRA-Gurobi M1 在 speed budget 内未产出 `tra_gurobi_s1_s9_summary.csv` 和可验收 Cmax。
- 失败原因记录为 `tra_gurobi_missing_or_nonfinite_cmax`。
- 因 TRA-Gurobi 未通过，TRA-Fast 未启动，M2-M9 未继续执行。

结果文件：

- `result/m_current_tra_acceptance_20260714/m_current_tra_acceptance_summary.csv`
- `result/m_current_tra_acceptance_20260714/m_current_tra_acceptance_summary.json`
- `docs/m_current_tra_acceptance_20260714.md`

