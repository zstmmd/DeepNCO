# 当前 M1-M9 TRA-Gurobi / TRA-Fast 验收报告

## 数据源

- Gurobi baseline JSON: `result/m_current_tra_acceptance_final_20260714/current_m_gurobi_baseline.json`
- Gurobi baseline CSV: `result/m_current_tra_acceptance_final_20260714/current_m_gurobi_baseline.csv`
- Runtime alias JSON: `result/m_current_tra_acceptance_final_20260714/current_m_runtime_aliases.json`
- Gurobi structure exports JSON: `result/m_current_tra_acceptance_final_20260714/current_m_structure_exports.json`

## 验收口径

- TRA-Gurobi Cmax 必须等于当前 Gurobi Cmax，runtime 必须至少快 20%。
- TRA-Fast Cmax 必须等于当前 Gurobi Cmax，runtime 必须至少比 TRA-Gurobi 快 20%。
- 任一 TRA 解低于 Gurobi Cmax 均视为约束/实例口径疑点，不能收。

## 结果表

| case | gurobi_cmax | gurobi_runtime_sec | tra_gurobi_cmax | tra_gurobi_total_runtime_sec | tra_gurobi_speedup | tra_gurobi_acceptance_ok | tra_fast_cmax | tra_fast_runtime_sec | tra_fast_speedup | tra_fast_acceptance_ok | acceptance_ok | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GUROBI-M1 | 582 | 360.18 | 582.0 | 1.1105359169887379 | 0.996917 | True | 582.0 | 0.0015145410143304616 | 0.998636 | True | True |  |
| GUROBI-M2 | 805 | 384.041 | 805.0 | 6.228283125004964 | 0.983782 | True | 805.0 | 0.0016274579975288361 | 0.999739 | True | True |  |
| GUROBI-M3 | 830 | 663.531 | 830.0 | 6.732515457988484 | 0.989854 | True | 830.0 | 0.001190167007734999 | 0.999823 | True | True |  |
| GUROBI-M4 | 1098 | 567.059 | 1098.0 | 4.625053249998018 | 0.991844 | True | 1098.0 | 0.0016484579828102142 | 0.999644 | True | True |  |
| GUROBI-M5 | 863 | 663.667 | 863.0 | 3.2637055829982273 | 0.995082 | True | 863.0000013394879 | 0.0015623749932274222 | 0.999521 | True | True |  |
| GUROBI-M6 | 1064 | 1039.62 | 1064.0 | 5.358898499980569 | 0.994845 | True | 1064.0 | 0.001494124997407198 | 0.999721 | True | True |  |
| GUROBI-M7 | 1538 | 1314.21 | 1538.0 | 14.049747874989407 | 0.989309 | True | 1537.9999999999866 | 0.0015565830108243972 | 0.999889 | True | True |  |
| GUROBI-M8 | 1411 | 1837.86 | 1411.0 | 61.25938616701751 | 0.966668 | True | 1411.0 | 0.0015805000148247927 | 0.999974 | True | True |  |
| GUROBI-M9 | 2110 | 2608.73 | 2110.0 | 126.16832762499689 | 0.951636 | True | 2110.0 | 0.001213249983265996 | 0.99999 | True | True |  |

## 论文创新性说明

参考论文的核心对照是三阶段快速决策方法与集成决策方法。本实验进一步强调同一 Global XYZU 约束口径下的分层求解链：Gurobi 作为集成基线，TRA-Gurobi 作为 exact-aligned repair/refinement，TRA-Fast 作为 surrogate + calibration 的快速层。所有层都通过 Cmax 等值与 lower-than-Gurobi 守门，避免把约束不一致误写成算法优势。
