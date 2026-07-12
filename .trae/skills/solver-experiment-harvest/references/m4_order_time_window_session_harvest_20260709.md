# M4 Order Time Window Session Harvest

本文件用于新会话迁移。适用 skill：`solver-experiment-harvest`。

## 新会话启动提示

新会话可直接说：

```text
Use Skill: solver-experiment-harvest
请读取 .trae/skills/solver-experiment-harvest/references/m4_order_time_window_session_harvest_20260709.md，
继续 M4 [22,16x5] + bq[3,3] + gap50 / gap4_qty34_tw180 的实验验收与下一轮设计。
```

优先证据路径：

- `test_gurobi_cut/time_window_simplification_report.md`
- `test_gurobi_cut/summary.csv`
- `test_gurobi_cut/results_800/tw_on_no_cut/tw_on_no_cut.metrics.json`
- `result/middle_bomseq_m4_seed42_t800_g01_sku22_16x5_bq33_gap50_noordertw_hit3_support20_copy1_stationtop1_slotlex_lbcuts_routearrlinear_r0_formal_20260702/gurobi_summary.json`
- 当前待进一步查看的打开文件：`result/middle_bomseq_m4_seed42_t800_g01_sku22_16x5_bq33_gap4_qty34_tw180_stationtop1_slotlex_lbcuts_routearrlinear_r0_formal_20260702/gurobi_summary.txt`

## 1. 实验意图

当前会话围绕 M4 Gurobi 中规模算例收敛与论文口径展开，核心问题是：

1. 当前 M4 最优配置 `[22,16x5] + bq[3,3] + gap50 + disable-order-time-windows` 是否可以进一步通过订单时间窗和模型简化策略改善收敛。
2. 订单时间窗引入是否会拖慢 Gurobi，还是会作为 tightening 改善收敛。
3. 五类简化策略是否值得作为下一轮正式模型改造方向：
   - 固定 station 选择
   - 限制 stack 候选数量
   - robot route 从 arc-level 改为 path/pattern-level
   - 固定或半固定 slot 顺序
   - SKU/tote 选择聚合到 stack 层

本轮不是普通参数 sweep，而是一次 solver experiment / case iteration 复盘，目标是判断哪些经验可沉淀到后续 M4/M-suite 调参。

## 2. 当前基线

### 2.1 用户给定 800s no-time-window 基线

配置：

```text
[22,16x5] + bq[3,3] + gap50 + disable-order-time-windows
```

指标：

| 指标 | 值 |
| --- | ---: |
| cmax | 884 |
| bound | 874.227 |
| gap | 1.195% |
| solve | 800.39s |
| vars | 22,993 |
| constr | 63,846 |
| u_arc | 17,531 |
| total_qty | 765 |
| span/deadline overrun | 0 / 0 |

对应结果目录：

```text
result/middle_bomseq_m4_seed42_t800_g01_sku22_16x5_bq33_gap50_noordertw_hit3_support20_copy1_stationtop1_slotlex_lbcuts_routearrlinear_r0_formal_20260702
```

### 2.2 目标约束

- M4 目标需要 `cmax > M3 baseline 827`。
- 尽量达到或接近 `gap <= 1%`。
- 中规模 Gurobi 求解时间应在 3600s 以内；当前关注 800s 内收敛质量。
- 默认不改地图；地图/layout/stack-grid 属于论文敏感配置。
- 保持密集存储口径：`tote_count / stack_count > 5`。
- 不随意改 BOM 数；BOM 数是论文链条敏感口径。

## 3. 本次变更

### 低风险变更

1. 在独立目录 `test_gurobi_cut/` 下创建实验资产，不污染主 solver：
   - `base_config_gap50.json`
   - `run_m4_time_window_cut_experiments.py`
   - `summarize_results.py`
   - `summary.csv`
   - `time_window_simplification_report.md`
2. 对同一 M4 gap50 配置开启/关闭 order time windows 做 120s 与 800s 对比。
3. 显式验证 station top1 与 station top2 负对照。
4. 显式验证 stack top2/top4。

### 中风险 / 只作 proxy 的变更

1. `s3_route_relaxed_proxy_tw` 使用 `integrate_u_route=False` 作为 route-pattern 思路的 proxy。
   - 注意：这不等价于真正 path/pattern formulation。
   - 只能说明“去掉 route arc 后规模上限能降多少”，不能作为正式模型验收。

### 高风险 / 未正式实现的变更

1. fixed-slot naive SKU chunk：
   - 使用 `GlobalXYZUConfig.fixed_work_units_by_order_slot` 做硬固定。
   - 结果 `WARM_START_FALLBACK`，说明 naive chunk 过约束，不可收。
2. stack-level SKU/tote aggregation：
   - 仅写了独立估算脚本 `experiments/estimate_stack_inventory_aggregation.py`。
   - 未改 `Gurobi/global_xyzu.py` 主模型。
   - 结论只能作为变量规模估算，不能当作求解验收。

## 4. 验收检查表

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| 规模是否递增 | 警惕 | 本轮主要是同一 M4 case 内部对比，不是 M-suite 链式递增验收。 |
| 中规模 Gurobi 是否 <= 3600s | 通过 | 800s case 均在 3600s 内。 |
| runtime 曲线是否合理 | 警惕 | 本轮是同规模策略对比，不用于证明 M-chain runtime 递增。 |
| tote/stack 是否 > 5 | 通过 | M4 配置 `tote=120`, `target_stack_count=20`, 比值约 6。 |
| 地图是否保持稳定 | 通过 | 使用 gap50 配置，未在本轮实验中额外改地图。 |
| warm start infeasible 是否已修复 | 警惕 | fixed-slot naive case 出现 `WARM_START_FALLBACK`，不能视为修复。其他主对比 case 未见该问题。 |
| 剪枝是否合理 | 通过/警惕 | station top1 合理；station top2 明显膨胀。route-relaxed proxy 不是合理正式剪枝，只能作 proxy。 |
| TRA-Gurobi 是否比 Gurobi 更快 | 证据不足 | 本轮没有跑 TRA-Gurobi / FixGurobi acceptance。 |
| TRA-Gurobi 是否与 Gurobi 解一致 | 证据不足 | 本轮没有 TRA-Gurobi 对齐实验。 |
| 是否符合论文口径 | 可保留但需带注释 | order time window 开启不破坏规模口径；route proxy、stack aggregation、naive fixed-slot 不可直接写成正式链条。 |

## 5. 关键结果

### 5.1 订单时间窗 800s 直接验证

开启 order time windows 后：

| 指标 | 值 |
| --- | ---: |
| cmax | 884 |
| bound | 874.222 |
| gap | 1.196% |
| solve | 800.27s |
| vars | 23,017 |
| constr | 63,927 |
| u_arc | 17,531 |
| span/deadline overrun | 0 / 0 |

相对 no-time-window 800s 基线：

- `vars +24`
- `constr +81`
- `u_arc` 不变
- `gap` 仅变化约 `+0.002` 个百分点
- `solve` 基本相同

结论：**订单时间窗没有拖慢 M4 gap50；在 120s 短时限下还明显改善 incumbent/gap。**

### 5.2 120s 筛选结果

| Case | cmax | bound | gap | solve | vars | constr | u_arc | overrun |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_no_tw | 967 | 874.219 | 9.722% | 120.12s | 22,993 | 63,846 | 17,531 | 0/0 |
| tw_on_no_cut | 895 | 874.219 | 2.434% | 120.12s | 23,017 | 63,927 | 17,531 | 0/0 |
| s1_station_fixed_top1_tw | 895 | 874.220 | 2.422% | 120.12s | 23,017 | 63,927 | 17,531 | 0/0 |
| s1_station_relaxed_top2_tw | 1229 | 874.095 | 63.216% | 120.03s | 44,082 | 126,942 | 37,966 | 229/44 |
| s2_stack_top2_tw | 895 | 874.220 | 2.433% | 120.11s | 23,017 | 63,927 | 17,531 | 0/0 |
| s2_stack_top4_tw | 895 | 874.220 | 2.433% | 120.13s | 23,017 | 63,927 | 17,531 | 0/0 |
| s3_route_relaxed_proxy_tw | 1217 | 766.769 | 12.988% | 120.02s | 4,044 | 7,481 | 0 | 0/0 |
| s4_fixed_slot_order_tw | 909 | NA | NA | 0.21s | 23,017 | 64,274 | 0 | 0/0 |

## 6. 踩坑点与根因

- 坑点：以为订单时间窗会拖慢模型。
  - 现象：开启 TW 后变量/约束增加。
  - 最可能根因：新增变量很少，只是 order-level arrival/span/deadline 变量和约束；同时 TW 给模型带来 tightening。
  - 证据：800s TW 与 no-TW 基本同 gap；120s TW gap 从 9.722% 降到 2.434%。
  - 是否已解决：已验证，TW 可保留。

- 坑点：stationtop2/stationall 看似增加选择自由，实际严重放大模型。
  - 现象：top2 vars 从 23,017 增到 44,082，u_arc 从 17,531 增到 37,966，gap 变成 63.216%，且出现 overrun。
  - 最可能根因：station 候选会放大 slot-station-rank 与 route task 组合。
  - 证据：`s1_station_relaxed_top2_tw` 结果。
  - 是否已解决：明确不建议放开 station top1。

- 坑点：stack top-k 继续收缩没有收益。
  - 现象：top2/top4 与 baseline TW 完全同规模。
  - 最可能根因：当前 colocated hit3/support20 已经让有效候选约每单 3 个 stack。
  - 证据：`s2_stack_top2_tw`、`s2_stack_top4_tw` 的 vars/constr/u_arc 与 TW baseline 相同。
  - 是否已解决：本 case 不再优先调 stack top-k。

- 坑点：route-relaxed proxy 不能替代 path-pattern formulation。
  - 现象：禁用 integrated route 后变量大幅下降，但 bound 掉到 766.769，gap 仍 12.988%。
  - 最可能根因：去掉 route arc 也去掉了关键 route 约束/下界表达。
  - 证据：`s3_route_relaxed_proxy_tw`。
  - 是否已解决：未解决；需要真正 path-pattern formulation。

- 坑点：naive fixed-slot 会过约束。
  - 现象：`s4_fixed_slot_order_tw` 返回 `WARM_START_FALLBACK`，无有效 Gurobi incumbent/bound。
  - 最可能根因：按 SKU chunk 固定 slot 与可行 warm route/station/stack 组合不一致。
  - 证据：status `WARM_START_FALLBACK`。
  - 是否已解决：未解决；后续必须从可行 warm solution 抽取固定/半固定结构。

- 坑点：`/usr/bin/python3` 没有 `gurobipy`。
  - 现象：smoke test 时报 `ModuleNotFoundError: No module named 'gurobipy'`。
  - 解决：使用 `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`。

- 坑点：`time-limit=1` 不等于总运行 1 秒。
  - 现象：time-limit=1 仍有十几秒。
  - 根因：SP4/warm start、建模、导出不受纯 Gurobi solve time 控制。
  - 是否已解决：已纳入实验解释。

## 7. 成功经验

### 可推广经验

1. 订单时间窗不一定是负担；如果只增加少量 order-level 变量和约束，可能作为有效 tightening。
2. station top1 是 M4 当前关键剪枝，放开 station 会成倍放大 route arc 和 station-rank 组合。
3. route_arc 是主瓶颈；真正值得做的是 path/pattern-level routing，而不是继续微调 SKU/tote 层。
4. 对“模型简化”要区分：
   - native config 可直接验收
   - proxy 只能定性
   - static estimate 不能当求解结果
5. 实验必须隔离目录，避免污染主 solver。`test_gurobi_cut/` 是可复用模式。

### 当前仅对本 case 有效

1. stack top2/top4 没收益，是因为当前 hit3/support20 和 colocated profile 已经把有效候选压小。
2. TW 在 120s 显著改善 gap，是 gap50 配置下的观察，迁移到其他 gap/qty/map 前需复验。
3. `gap50` no-TW 与 TW 在 800s 几乎等价，不能直接推断所有 M4 配置都等价。

## 8. 下一轮建议

### P0：检查当前打开的 `gap4_qty34_tw180` 结果

当前 IDE 打开的文件：

```text
result/middle_bomseq_m4_seed42_t800_g01_sku22_16x5_bq33_gap4_qty34_tw180_stationtop1_slotlex_lbcuts_routearrlinear_r0_formal_20260702/gurobi_summary.txt
```

已知打开行：

```text
task_count=19
```

建议新会话优先读取其 `gurobi_summary.json/txt`，判断：

- cmax 是否仍 >827
- gap 是否优于 gap50
- total_qty 是否变化为 qty34 预期
- TW180 是否带来 overrun
- vars/constr/u_arc 是否仍保持可比
- 是否比 gap50 发生口径漂移

风险：`gap4_qty34_tw180` 可能通过 qty/window 改动改变问题本质，需按 solver-experiment-harvest 验收表判断。

### P1：保持 TW enabled + stationtop1，做 gap/qty/window 局部对比

建议动作：

- 对比 `gap50 + TW on`、`gap4_qty34_tw180`、已有 `gap10/gap22/gap30` 结果。
- 只读取结果，不先开新长跑。

预计收益：

- 快速判断下一步是否应围绕 `gap4_qty34_tw180` 作为新候选基线。

风险：

- 如果 qty/window 改动过多，不能直接和 gap50 基线横比。

是否影响论文口径：

- gap/window 属于地图/时间结构口径，应带注释；qty 改动会影响需求强度，也需标注。

### P2：设计真正 path-pattern route formulation

建议动作：

- 先从 SP4/warm solution 生成每 robot 少量候选 route pattern。
- Gurobi 主模型只选择 pattern 或做局部 arc repair。
- 保留 route lower bound，避免像 route-relaxed proxy 那样 bound 过弱。

预计收益：

- 直接针对 `route_arc=17531` 主瓶颈。

风险：

- 需要新 formulation 和可行性后处理；短期不能直接纳入正式论文链条。

是否影响论文口径：

- 算法层创新/加速可解释，但必须保证与 Gurobi full model 解一致或在 acceptance 容忍范围内。

### P3：fixed-slot 只能基于可行 warm solution 半固定

建议动作：

- 从已验证可行 warm solution 中抽取 slot assignment。
- 允许局部 swap，不要按 SKU 顺序硬 chunk。

预计收益：

- 可能减少搜索分支。

风险：

- 过约束导致 infeasible / fallback。

是否影响论文口径：

- 若作为 heuristic warm/fix 策略，需要明确与 full Gurobi 的解质量一致性。

## 9. 一句话结论

**本轮 gap50 + order time windows 实验可保留但需带注释：订单时间窗对 M4 gap50 不构成求解速度负担，stationtop1 是必须保留的稳定剪枝；但 route proxy、naive fixed-slot、stack 聚合都不能收为正式链条，下一轮应优先检查 `gap4_qty34_tw180` 是否是更强候选，并把真正 path-pattern routing 作为主要模型改造方向。**

