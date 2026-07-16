# 当前 M1-M9 TRA-Gurobi / TRA-Fast 验收报告

## 数据源

- Gurobi baseline JSON: `result\m_current_tra_formal_m1_warm270_v2_retry_20260716\current_m_gurobi_baseline.json`
- Gurobi baseline CSV: `result\m_current_tra_formal_m1_warm270_v2_retry_20260716\current_m_gurobi_baseline.csv`
- Runtime alias JSON: `result\m_current_tra_formal_m1_warm270_v2_retry_20260716\current_m_runtime_aliases.json`
- Gurobi structure exports JSON: `result\m_current_tra_formal_m1_warm270_v2_retry_20260716\current_m_structure_exports.json`

## 验收口径

- TRA-Gurobi Cmax 必须等于当前 Gurobi Cmax，runtime 必须至少快 20%。
- TRA-Fast Cmax 必须等于当前 Gurobi Cmax，runtime 必须至少比 TRA-Gurobi 快 20%。
- 任一 TRA 解低于 Gurobi Cmax 均视为约束/实例口径疑点，不能收。

## 结果表

| case | gurobi_cmax | gurobi_runtime_sec | tra_gurobi_cmax | tra_gurobi_total_runtime_sec | tra_gurobi_speedup | tra_gurobi_acceptance_ok | tra_fast_cmax | tra_fast_runtime_sec | tra_fast_speedup | tra_fast_acceptance_ok | acceptance_ok | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GUROBI-M1 | 582 | 360.18 | inf | 306.67688689986244 | nan | False |  |  |  |  | False | tra_gurobi_missing_or_nonfinite_cmax |

## 论文创新性说明

参考论文的核心对照是三阶段快速决策方法与集成决策方法。本实验进一步强调同一 Global XYZU 约束口径下的分层求解链：Gurobi 作为集成基线，TRA-Gurobi 作为 exact-aligned repair/refinement，TRA-Fast 作为 surrogate + calibration 的快速层。所有层都通过 Cmax 等值与 lower-than-Gurobi 守门，避免把约束不一致误写成算法优势。
