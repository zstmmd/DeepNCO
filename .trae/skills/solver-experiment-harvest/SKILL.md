---
name: "solver-experiment-harvest"
description: "Harvests RMFS solver experiment progress, validates thesis-safe acceptance, and extracts reusable lessons. Invoke when iterating Gurobi/TRA-Gurobi cases, debugging mismatch, or proposing next scales."
---

# Solver Experiment Harvest

用于 `deepnco` 研究中的 **算例迭代、实验复盘、验收判断、踩坑归因、成功经验沉淀**。

这个 skill 的中心对象不是普通对话，而是 **一次 solver experiment / case iteration**。
当用户在做 Gurobi / TRA-Gurobi / FixGurobi / TRA-Fast 的实验推进时，优先使用本 skill。

## Invoke When

出现以下任一场景时调用：

- 用户说“帮我总结这次算例尝试”
- 用户说“看看 M3/M4/M5 下一步怎么调”
- 用户说“这次 Gurobi/TRA-Gurobi 结果能不能收”
- 用户说“总结踩坑点 / 成功经验 / 下次应该试什么”
- 用户说“两个算法为什么没对齐”
- 用户说“warm start 不可行怎么修”
- 用户说“这个剪枝是不是不合理”
- 用户说“要保证论文口径，帮我判断哪些配置不能改”

## Target Project

默认代码仓：

```text
/Users/bytedance/Documents/trae_projects/deepnco
```

默认研究文档：

```text
当前研究进展：
https://my.feishu.cn/wiki/M2lawvFdviVF4ckdP3Wck14ln4b

参考论文：
https://bytedance.larkoffice.com/docx/RQqldCb6boKXV2xjwI7cfAUInQf
```

## Core Goal

针对每次实验，回答 4 个核心问题：

1. **这次到底改了什么？**
2. **这次结果相对上一版是进步、退步，还是口径漂移？**
3. **这次结果能不能作为论文链条中的正式算例保留？**
4. **可复用的成功经验是什么，下一个最优实验是什么？**

## Hard Constraints

以下约束默认视为 **硬约束**。若用户没有明确推翻，必须严格遵守：

1. **规模递增**
   - 算例链应保持规模递增。
   - 中规模链条下，BOM 数、SKU 数、相关资源规模应按研究口径递增或至少不倒退。

2. **中规模 Gurobi 时间目标**
   - 中规模 Gurobi 求解时间目标为 **3600s 以内**。
   - 最好也保持随规模递增而 **整体不逆序**。
   - 若出现更大算例反而明显更快，默认视为异常，必须解释原因。

3. **BOM 数由用户预设**
   - 不擅自改 BOM 数口径。
   - 若确需改，必须明确说明会破坏当前论文链条。

4. **Warm Start 不可行必须修复**
   - 不允许把 warm start infeasible 当作“可忽略”。
   - 必须定位并说明：是初始解构造问题、约束不对齐、候选集剪掉了必要变量，还是后处理不一致。

5. **尽量不修改地图配置**
   - 地图 / layout / stack-grid 属于论文敏感基础配置。
   - 若没有强证据，不建议改地图。
   - 对“密集存储”场景，默认要求：
     - `tote_count / stack_count > 5`
   - 若不满足，默认判为不符合场景口径。

6. **不能比上一个算例好太多**
   - 若新算例规模更大，却在 Cmax、runtime、gap 等维度异常地“轻松更优”，默认怀疑：
     - 剪枝过强
     - 约束缺失
     - 候选集收缩改变了问题本质
     - warm-start / validation 口径不一致

7. **剪枝策略必须合理**
   - 若剪掉了本该保留的可行域，导致：
     - Gurobi 解比 TRA-Gurobi 更差
     - 或 TRA-Gurobi 比 Gurobi 更优
   - 默认优先怀疑模型不一致，而不是直接接受结果。

8. **TRA-Gurobi 必须比 Gurobi 快，且解一致**
   - 默认目标：
     - `TRA runtime < Gurobi runtime`
     - `TRA objective / Cmax == Gurobi objective / Cmax`
   - 若不一致，必须明确说明是否在“论文允许容忍范围”内；若没有明确容忍口径，默认按“不通过”处理。

9. **与论文口径一致**
   - 算例规模、求解速度、实验组织方式尽量与论文中的实验故事线一致。
   - 若为了收敛而引入会被论文答辩挑战的配置改动，必须显式标红。

## Repo Evidence Sources

优先阅读以下证据源：

### 文档 / 结果

- `docs/medium_scale_results_20260616.md`
- `experiments/calibrate_gurobi_m_suite.md`
- 研究进展飞书文档中的 M-suite / M3 / M4 / acceptance 内容

### 实验脚本

- `experiments/calibrate_gurobi_m_suite.py`
- `experiments/run_m_layout_acceptance_suite.py`
- `experiments/run_m_tra_regression.py`
- `experiments/run_tra_fixgurobi_budgeted_suite.py`
- `experiments/summarize_tra_gurobi_acceptance.py`
- `experiments/run_gurobi_scale_suite.py`
- `experiments/run_large_scale_gurobi.py`

### 模型与求解核心

- `Gurobi/global_xyzu.py`
- `Gurobi/tra_gurobi.py`
- `Gurobi/sp4.py`

## Companion Skill Boundary

仓内已有：

```text
.trae/skills/stacked-runtime-calibration/SKILL.md
```

两者边界如下：

- **`stacked-runtime-calibration`**
  - 专注 `STACK-S1..S9`
  - 关注 stacked runtime 标定、剪枝、Cmax、gap、保留结果目录
- **`solver-experiment-harvest`**
  - 专注 **实验验收、论文口径、踩坑归因、成功经验沉淀**
  - 覆盖 `M-suite`、`Gurobi vs TRA-Gurobi`、`warm start infeasible`、约束一致性

如果用户只是要“继续调 S1-S9 参数”，优先用 `stacked-runtime-calibration`。  
如果用户要“判断实验能不能收、为什么不收、下一轮怎么试”，优先用本 skill。

## Repository-Native Evidence Precedence

同一 case 存在多个结果来源时，按以下优先级取证：

1. **最终 acceptance 汇总**
   - `m_layout_acceptance_summary.csv`
   - 适用于 M-suite 的最终“是否收”判断
2. **TRA-Gurobi 对比汇总**
   - `tra_gurobi_acceptance_summary.csv`
   - `tra_gurobi_s1_s9_summary.csv`
   - 适用于 Gurobi / TRA-Gurobi 的速度与质量对齐判断
3. **单次 Gurobi 结果**
   - `summary.csv`
   - `run_details.json`
4. **研究文档 / markdown 报告**
   - `docs/*.md`
   - 飞书研究进展文档

若不同来源冲突，默认：

- **最终 acceptance 汇总 > 单次运行细节 > 手工文档**
- 但若 acceptance 汇总明显过旧，应明确标注“汇总已过期，以最新 run 为准”

## Repository-Native Acceptance Contract

对 `M-suite`，优先遵循仓内 `experiments/run_m_layout_acceptance_suite.py` 的验收口径，而不是临时自创标准。

当前脚本里的关键 acceptance 字段包括：

- `gurobi_gap_ok`
- `fix_quality_ok`
- `fix_runtime_ok`
- `fast_quality_ok`
- `fast_runtime_ok`
- `cmax_gt_s9_ok`
- `cmax_increasing_ok`
- `runtime_scale_ok`
- `acceptance_ok`

其中 `acceptance_ok` 的含义是：**上述关键检查全部为 true**。

默认阈值来自脚本参数：

- `gurobi_mip_gap = 0.01`
- `tra_fix_cmax_abs_tol = 3.0`
- `tra_fix_runtime_cap_sec = 1600.0`
- `tra_fast_runtime_cap_sec = 300.0`
- `tra_fast_gap_cap = 0.03`
- `min_runtime_ratio_vs_previous = 0.70`
- `min_cmax = 438.0`

解释规则：

1. **Gurobi 侧**
   - `gurobi_gap_ok=true` 才说明本轮 Gurobi 自身达到了仓内默认 gap 口径。

2. **TRA-FixGurobi 侧**
   - `fix_quality_ok=true` 表示与 Gurobi 的 cmax 偏差在容忍阈值内。
   - `fix_runtime_ok=true` 表示 exact repair 时间没有超出当前仓内 cap。

3. **TRA-Fast 侧**
   - `fast_quality_ok=true` 表示 speed baseline 的质量偏差仍在容忍范围。
   - `fast_runtime_ok=true` 表示 speed baseline 的速度符合当前仓内 cap。

4. **链式规模验收**
   - `cmax_increasing_ok=true`：说明 cmax 没有相对前一个 case 逆序。
   - `runtime_scale_ok=true`：说明 runtime 没有异常快到破坏规模故事线。
   - `cmax_gt_s9_ok=true`：当前链条至少没有比既有尾部基线更弱。

如果用户问“这次能不能收”，而仓内已经产出 `m_layout_acceptance_summary.csv`，优先直接依据该文件回答。  
如果没有该文件，再退回到 `summary.csv + run_details.json + tra_gurobi_s1_s9_summary.csv` 进行手工判定。

## Calibration Contract For M-Suite

对 `M1..M9` 的 Gurobi 标定，优先遵循 `experiments/calibrate_gurobi_m_suite.md` 和对应脚本口径：

- 问题规模至少有一个维度增加，且其余关键维度不倒退
- `model_cmax` 递增
- wall-clock runtime 递增
- `model_gap <= 0.01`
- runtime 不超过 time limit

当前固定 route pruning policy：

- `route_arc_prune=True`
- `enable_route_load_interval_arc_prune=True`
- `enable_route_time_window_arc_prune=False`
- `enable_route_directional_arc_prune=False`
- `route_pickup_neighbor_limit=0`
- warm start disabled
- scale-adaptive candidate prune disabled

因此如果本轮试验违反上述固定策略，默认要在结论里显式说明：

- 这是“偏离 calibrated baseline”的实验
- 不能直接和正式 M-chain 横向对比

## Result Files And Trusted Fields

做实验复盘时，优先读取以下文件，并使用对应字段：

### 1. `summary.csv`

重点字段：

- `status`
- `model_cmax`
- `runtime_sec`
- `model_gap`
- `model_best_bound`

### 2. `run_details.json`

重点字段：

- `model_var_count_total`
- `u_arc_count`
- `candidate stack / station / route` 相关计数
- 运行参数回显

### 3. `tra_gurobi_s1_s9_summary.csv`

重点字段：

- `tra_gurobi_cmax`
- `tra_gurobi_total_runtime_sec`
- `tra_gurobi_time_to_optimal_sec`
- `gap_vs_gurobi_pct`
- `known_target_guidance`
- `global_target_probe_enabled`

### 4. `tra_gurobi_acceptance_summary.csv`

重点字段：

- `runtime_pass`
- `quality_pass`
- `optimal_pass`
- `acceptance_pass`

### 5. `m_layout_acceptance_summary.csv`

重点字段：

- `acceptance_ok`
- `gurobi_gap_ok`
- `fix_quality_ok`
- `fix_runtime_ok`
- `fast_quality_ok`
- `fast_runtime_ok`
- `cmax_increasing_ok`
- `runtime_scale_ok`

## Warm-Start Infeasible Handling

只要出现 warm start infeasible，必须额外追问和定位：

1. 是否是初始解构造错误
2. 是否是 Gurobi / TRA-Gurobi 约束集合不一致
3. 是否是 candidate stack / station / route 剪枝剪掉了 warm start 依赖变量
4. 是否是 SP4 / route 子问题不可行
5. 是否只是 fallback 成功，但根因未修

若日志中只看到“fallback 成功”，而没有根因修复证据，结论必须写：

```text
warm start 不可行已被绕过，但未被真正修复，不能视为口径稳定。
```

## Preferred Investigation Order

做判断时，按以下顺序推进，不要一上来就拍脑袋改参数：

1. **先认口径**
   - 当前 case 在实例链中的位置是什么
   - 上一版正式基线是什么
   - 用户规定的 BOM / SKU / stack / tote / map 口径是什么

2. **再认变更**
   - 本次相对上一版改了哪些参数
   - 是规模变化、约束变化、剪枝变化、warm-start 变化，还是求解器参数变化

3. **再认结果**
   - Gurobi 的 cmax / bound / gap / runtime / vars / constr
   - TRA-Gurobi 的 cmax / runtime / gap_vs_gurobi
   - 是否通过 acceptance

4. **最后再归因**
   - 结果变化更像是规模上升带来的自然变慢
   - 还是剪枝/约束/候选集口径漂移导致的“假进步”或“假退步”

## Decision Ladder

当用户问“下一步该怎么调”，优先从 **低风险改动** 到 **高风险改动** 排序：

1. **先修一致性**
   - warm start infeasible
   - Gurobi 与 TRA-Gurobi 约束不一致
   - acceptance 脚本口径不一致

2. **再修剪枝**
   - candidate stack / station topk
   - route pickup neighbor
   - 线性化 / 下界 cut
   - 是否出现过强剪枝导致的问题本质改变

3. **再调求解参数**
   - time limit
   - mip gap
   - focus / cuts / presolve 类参数

4. **再调实例规模**
   - BOM / SKU / tote / stack / batch qty 的递增链

5. **地图改动最后再考虑**
   - 如果要改地图，必须说明：
     - 为什么现有地图已无法支撑目标
     - 为什么这个改动不会破坏论文口径

## Common Pitfalls Library

每次总结时，重点检查以下常见坑：

1. **Gurobi 剪枝后解比 TRA-Gurobi 还差**
   - 默认怀疑剪枝过强或约束错位。

2. **TRA-Gurobi 比 Gurobi 更优**
   - 默认不是“TRA 更厉害”，而是两个模型不对齐。

3. **Warm start infeasible 被静默绕过**
   - 这是隐患，不是修复。

4. **更大实例反而快太多**
   - 默认怀疑案例结构变了，不是算法真的变强了。

5. **地图或基础配置被悄悄改动**
   - 容易破坏论文说服力。

6. **密集存储特征被破坏**
   - `tote_count / stack_count <= 5` 时要明确预警。

7. **候选集裁剪改变问题本质**
   - 表面更快，但不再是同一问题。

8. **与论文结果曲线不一致**
   - 包括规模曲线、runtime 曲线、acceptance 口径曲线。

## Output Contract

默认输出必须包含以下 8 段，除非用户明确要求简版：

### 1. 实验意图

- 当前要推进哪个 case
- 当前卡在哪个问题
- 本轮实验的主要目标

### 2. 当前基线

- 上一个可接受 case 是什么
- 关键规模参数
- Gurobi / TRA-Gurobi 的基线表现

### 3. 本次变更

- 这次改了哪些参数 / 约束 / 剪枝 / warm-start
- 按“低风险 / 高风险”标注

### 4. 验收检查表

必须输出如下检查项，并给出 `通过 / 警惕 / 不通过`：

- 规模是否递增
- 中规模 Gurobi 是否 <= 3600s
- runtime 曲线是否合理
- tote/stack 是否 > 5
- 地图是否保持稳定
- warm start infeasible 是否已修复
- 剪枝是否合理
- TRA-Gurobi 是否比 Gurobi 更快
- TRA-Gurobi 是否与 Gurobi 解一致
- 是否符合论文口径

### 5. 踩坑点与根因

至少按以下格式输出：

```markdown
- 坑点：
  - 现象：
  - 最可能根因：
  - 证据：
  - 是否已解决：
```

### 6. 成功经验

只记录 **可复用** 的经验，不记录偶然结果。

必须区分：

- **可推广经验**
- **当前仅对本 case 有效**

### 7. 下一轮最优实验建议

给出 1 到 3 个候选动作，并按优先级排序：

- 建议动作
- 预计收益
- 风险
- 是否影响论文口径

### 8. 一句话结论

必须用一句话明确给出：

- `可收为正式链条`
- `可保留但需带注释`
- `不能收，优先修一致性`
- `不能收，问题定义已漂移`

## Preferred Output Shape

优先使用下面这个格式：

```markdown
## 本次实验结论

### 1. 实验意图

### 2. 当前基线

### 3. 本次变更

### 4. 验收检查表
| 检查项 | 结果 | 说明 |
| --- | --- | --- |

### 5. 踩坑点与根因

### 6. 成功经验

### 7. 下一轮建议

### 8. 一句话结论
```

若用户希望“按固定复盘模板输出”，优先参照：

```text
templates/experiment_retrospective_template.md
```

除非用户明确要求更短版本，否则应尽量填满该模板中的关键字段。

## Non-Negotiable Rules

- **不要**把“更快”直接等同于“更好”
- **不要**把“更优 cmax”直接等同于“算法更强”
- **不要**在约束没对齐时比较算法优劣
- **不要**为了过实验而轻易改地图基础配置
- **不要**忽略 warm start infeasible
- **不要**在没有证据时把偶然成功写成经验

## When Evidence Is Missing

如果证据不够，明确指出缺什么，不要假装知道：

- 缺上一版基线
- 缺 Gurobi summary / acceptance summary
- 缺 warm-start infeasible 日志
- 缺候选集 / 剪枝参数
- 缺论文对应口径

当证据缺失时，优先输出：

1. 当前能确认的事实
2. 当前不能确认的部分
3. 下一步需要补的最小证据集

## Recommended Language

默认用中文输出，语气偏研究复盘，不写空话。

优先使用：

- “口径”
- “基线”
- “验收”
- “一致性”
- “剪枝合理性”
- “论文可辩护性”

避免使用：

- “看起来不错”
- “应该没问题”
- “大概率”

没有证据就直接说“证据不足”。
