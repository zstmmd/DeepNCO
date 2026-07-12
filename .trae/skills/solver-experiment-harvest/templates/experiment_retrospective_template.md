# 本次实验结论

## 1. 实验意图

- 当前 case：
- 当前链条位置：
- 本轮主要目标：
- 是否属于正式论文链条实验：

## 2. 当前基线

- 上一个可接受 case：
- 上一版 Gurobi：
  - cmax：
  - runtime_sec：
  - gap：
- 上一版 TRA-FixGurobi：
  - cmax：
  - runtime_sec：
- 上一版 TRA-Fast：
  - cmax：
  - runtime_sec：

## 3. 本次变更

| 变更项 | 本次值 | 上次值 | 风险等级 | 备注 |
| --- | --- | --- | --- | --- |
| BOM数 |  |  | 低/中/高 |  |
| SKU数 |  |  | 低/中/高 |  |
| Tote数 |  |  | 低/中/高 |  |
| Stack数 |  |  | 低/中/高 |  |
| 地图/布局 |  |  | 低/中/高 |  |
| candidate stack |  |  | 低/中/高 |  |
| station topk |  |  | 低/中/高 |  |
| route pickup neighbor |  |  | 低/中/高 |  |
| warm start |  |  | 低/中/高 |  |
| gap / time limit |  |  | 低/中/高 |  |

## 4. 结果摘要

### 4.1 Gurobi

- status：
- model_cmax：
- model_best_bound：
- model_gap：
- runtime_sec：
- model_var_count_total：
- u_arc_count：

### 4.2 TRA-FixGurobi

- tra_gurobi_cmax：
- tra_gurobi_total_runtime_sec：
- gap_vs_gurobi_pct：

### 4.3 TRA-Fast

- tra_fast_cmax：
- tra_fast_runtime_sec：
- tra_fast_vs_gurobi_gap：

## 5. 验收检查表

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| 规模是否递增 | 通过/警惕/不通过 |  |
| BOM / SKU / Tote / Stack 是否未倒退 | 通过/警惕/不通过 |  |
| 中规模 Gurobi 是否 <= 3600s | 通过/警惕/不通过 |  |
| runtime 曲线是否合理 | 通过/警惕/不通过 |  |
| tote/stack 是否 > 5 | 通过/警惕/不通过 |  |
| 地图是否保持稳定 | 通过/警惕/不通过 |  |
| warm start infeasible 是否已真正修复 | 通过/警惕/不通过 |  |
| 剪枝是否合理 | 通过/警惕/不通过 |  |
| TRA-FixGurobi 是否与 Gurobi 一致 | 通过/警惕/不通过 |  |
| TRA-FixGurobi 是否快于 Gurobi | 通过/警惕/不通过 |  |
| TRA-Fast 是否在速度口径内 | 通过/警惕/不通过 |  |
| 是否符合论文口径 | 通过/警惕/不通过 |  |

### 5.1 仓内 acceptance 字段

- acceptance_ok：
- gurobi_gap_ok：
- fix_quality_ok：
- fix_runtime_ok：
- fast_quality_ok：
- fast_runtime_ok：
- cmax_gt_s9_ok：
- cmax_increasing_ok：
- runtime_scale_ok：

## 6. 踩坑点与根因

- 坑点 1：
  - 现象：
  - 最可能根因：
  - 证据：
  - 是否已解决：

- 坑点 2：
  - 现象：
  - 最可能根因：
  - 证据：
  - 是否已解决：

## 7. 成功经验

### 7.1 可推广经验

- 

### 7.2 当前仅对本 case 有效

- 

## 8. 下一轮最优实验建议

| 优先级 | 建议动作 | 预计收益 | 风险 | 是否影响论文口径 |
| --- | --- | --- | --- | --- |
| P1 |  |  |  | 是/否 |
| P2 |  |  |  | 是/否 |
| P3 |  |  |  | 是/否 |

## 9. 一句话结论

- 可收为正式链条 / 可保留但需带注释 / 不能收，优先修一致性 / 不能收，问题定义已漂移

## 10. 最小补证清单

- 缺失证据 1：
- 缺失证据 2：
- 缺失证据 3：
