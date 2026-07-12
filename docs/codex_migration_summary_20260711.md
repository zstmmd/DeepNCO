# Codex 迁移归纳 - deepnco

日期：2026-07-11

本文件用于把当前 `deepnco` 项目的结构、实验结论、已探测参数、关键文件和 Codex 迁移入口整理成一个可接手版本。当前项目主线是密集堆垛式 RMFS/OFS 算例的 Gurobi / TRA-Gurobi 求解与论文实验口径校准。

## 1. 项目结构

核心代码目录：

| 路径 | 作用 | 迁移优先级 |
|---|---|---:|
| `Gurobi/global_xyzu.py` | 当前最重要的 Global XYZU Gurobi MILP 求解器。支持 warm start、route arc、lazy constraint、route/time-window prune、下界 cuts。 | 必迁 |
| `Gurobi/tra_gurobi.py` | TRA-Gurobi / FixGurobi 相关入口。用于把启发式/TRA 解和 Gurobi 精修对齐。 | 必迁 |
| `Gurobi/sp1.py`..`sp4.py` | 分阶段构造：订单拆分、站台分配、任务排序、路径/机器人路由。warm start 依赖这些阶段。 | 必迁 |
| `Gurobi/resource_time_alns/` | resource-time ALNS / repair / validator / diagnostics。TRA-Fast、FixGurobi 证据依赖这里。 | 必迁 |
| `problemDto/createInstance.py` | 算例生成入口；读取 runtime config 后生成 STACK/M 系列问题实例。 | 必迁 |
| `entity/` | robot、tote、stack、station、order、task 等基础实体。 | 必迁 |
| `config/ofs_config.py` | 关键时间常数：拣选、搬运、kit window、BOM arrival window 等。 | 必迁 |
| `experiments/` | 所有实验驱动脚本、汇总脚本、校准脚本。 | 必迁 |
| `tests/` 与根目录 `test_*.py` | 基础回归和候选池/运行时优化测试。 | 建议迁 |
| `docs/` | 已整理的论文实验表、参数表、迁移说明。 | 必迁 |
| `result/` | 实验输出，体积大；只迁移最终接受和当前探测证据。 | 选择迁 |
| `diagrams/` | 可视化输出，非求解必需。 | 可选 |

运行环境：

- Python 3.12 系列环境；仓内有 `environment.yml`，但内容较大，需要在 Codex 侧确认是否可直接创建。
- 必需商业/本地依赖：`gurobipy` 与可用 Gurobi license。
- 部分 warm start / SP4 可能依赖 OR-Tools；若没有 OR-Tools，应确认 `--disable-warm-start-sp4` 或 fallback 行为。

## 2. 主要实验线

当前有两条应分开迁移的实验线。

### 2.1 STACK-S1 到 STACK-S9

定位：小到中等规模的单 block 密集堆垛 benchmark。Codex 迁移主口径应使用 **Gurobi-vs-TRA 对齐表**；`stacked_single_block_runtime_report.md` 是另一套 runtime calibration 参考表，不能混用。

当前接受文件：

- 配置：`experiments/configs/stacked_single_block_runtime_configs.json`
- 报告：`experiments/configs/stacked_single_block_runtime_report.md`
- 历史 Gurobi 结果表：`docs/stacks_s1_s9_gurobi_results_20260621.md`
- Gurobi 侧嵌入字段汇总：`result/stacks_s1_s9_embedded_fields_20260622/summary.csv`
- TRA 侧对齐证据：分散在 `result/task18_*/*/tra_gurobi_s1_s9_summary.{csv,txt}`，例如 S4/S5/S7/S9 的 TRA runtime 行。

runtime calibration 保留结果目录：

```text
result/stacked_single_block_runtime_s1_range_1_5_batch_1_3_station2_pruned_r0
result/stacked_single_block_runtime_s2_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s3_range_1_5_batch_1_3_station1_no_path_prune
result/stacked_single_block_runtime_s4_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s5_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s6_range_1_5_batch_1_3_station2_pruned_r0
result/stacked_single_block_runtime_s7_baseline_gurobi_log_t220
result/stacked_single_block_runtime_s8_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s9_seed42_qty_1_5_batch_3_5_stack211111_cand7_station2_r5_t260
```

迁移主表：

| Case | Gurobi UB | Gurobi LB | Gurobi gap | TRA Cmax | Gurobi Runtime | TRA Runtime | Cmax gap | Time gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STACK-S1 | 103 | 102.641 | 0.35% | 103 | 7.50 | 3.628 | 0.00% | -51.64% |
| STACK-S2 | 158 | 157.056359 | 0.60% | 158 | 20.42 | 10.63 | 0.00% | -47.95% |
| STACK-S3 | 239 | 238.952 | 0.02% | 239 | 28.19 | 17.09 | 0.00% | -39.37% |
| STACK-S4 | 256 | 256 | 0.00% | 256 | 38.72 | 22.096 | 0.00% | -42.93% |
| STACK-S5 | 269 | 266.54235 | 0.92% | 269 | 41.91 | 24.016 | 0.00% | -42.69% |
| STACK-S6 | 389 | 383.54063 | 1.42% | 389 | 150.08 | 82.841 | 0.00% | -44.80% |
| STACK-S7 | 524 | 520.06 | 0.76% | 524 | 150.14 | 83.805 | 0.00% | -44.18% |
| STACK-S8 | 658 | 653.56 | 0.68% | 658 | 127.11 | 56.48 | 0.00% | -55.57% |
| STACK-S9 | 781 | 772.092105 | 1.15% | 781 | 260.12 | 142.088 | 0.00% | -45.38% |

核心结论：

- 固定 `seed=42`。
- 固定 dense stacked layout，`target_stack_count=8`，平均 tote/stack 大于 `5.5`。
- 主表 Cmax 单调递增：`103 -> 158 -> 239 -> 256 -> 269 -> 389 -> 524 -> 658 -> 781`。
- TRA 与 Gurobi 的 Cmax 完全对齐，`cmax gap=0%`。
- TRA runtime 全部更快，降幅约 `39% - 56%`。
- S6 与 S9 的 Gurobi gap 高于 1%，作为 accepted-with-note 结果使用。
- `stacks_s1_s9_embedded_fields_20260622/summary.csv` 与主表不完全一致，例如 S4/S6/S7 的 Cmax 口径不同；迁移时以本节主表为最终呈现口径。
- `stacked_single_block_runtime_report.md` 中另一套 calibration 表仍可用于复现实验，但不要覆盖本节 Gurobi-vs-TRA 对齐主表。

runtime calibration 复现入口示例：

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py \
  --scales STACK-S9 \
  --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json \
  --seed 42 \
  --time-limit 260 \
  --mip-gap 0.01 \
  --candidate-stack-topk 7 \
  --candidate-station-topk-per-stack 2 \
  --route-pickup-neighbor-limit 5 \
  --disable-warm-start \
  --disable-warm-start-sp4 \
  --output-dir result/stacked_single_block_runtime_s9_seed42_qty_1_5_batch_3_5_stack211111_cand7_station2_r5_t260
```

### 2.2 中规模 M-suite

定位：M1 到 M9 的中规模论文链条，用于证明更大结构化实例下的扩展能力。

当前相关文件：

- 旧结果概览：`docs/medium_scale_results_20260616.md`
- 参数说明：`docs/middle_m1_m9_parameters_for_lark.xml`
- Gurobi calibration 说明：`experiments/calibrate_gurobi_m_suite.md`
- 旧主配置：`experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json`
- M4 大量探测配置：`experiments/configs/m4_sku22x6_bq33_gap4_qty34_tw180*.json`

重要注意：

- `experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json` 中 M7 仍是旧的 `5 robots / 2 stations` 口径。
- 最新已达标 M7 是 `5 robots / 3 stations` 的 `r5s3` 版本，结果目录已经存在，但未发现已固化回主 `experiments/configs` 的同名配置文件。
- Codex 迁移时不能只看旧 JSON；需要把最新 M7 作为当前工作状态单独带过去。

#### 当前 M1-M6 主表

以下表是当前中规模迁移主证据，六个 `gurobi_summary.json` 路径均已在本地存在并核对。

| Case | 算例名称 | 上界 Cmax | 下界 BestBound | Gap | Runtime(s) | Status | Vars / Constr | 证据路径 |
|---|---|---:|---:|---:|---:|---|---:|---|
| M1 | `middle_bomseq_m1_seed42_t360_g002_nocand_routeprune_r0_20260624` | 582.208 (Cmax 582) | 579.06 | 0.541% | 360.180 | TIME_LIMIT | 17482 / 48246 | `result/middle_bomseq_m1_seed42_t360_g002_nocand_routeprune_r0_20260624/gurobi_summary.json` |
| M2 | `middle_bomseq_m2_seed42_t400_g01_authoritative_noslotlex_focus1_h095_r0_probe_20260709` | 805 | 798.09 | 0.907% | 384.0 | OPTIMAL | 35573 / 101351 | `result/middle_bomseq_m2_seed42_t400_g01_authoritative_noslotlex_focus1_h095_r0_probe_20260709/gurobi_summary.json` |
| M3 | `middle_bomseq_m3_seed42_t700_g01_4x5_r4s3_t115_sku270_bq33_chunked4_stationtop1_routearrlinear_noslotlex_focus1_h095_r0_probe_20260709` | 830 | 822.09 | 0.985% | 663.5 | OPTIMAL | 30483 / 84200 | `result/middle_bomseq_m3_seed42_t700_g01_4x5_r4s3_t115_sku270_bq33_chunked4_stationtop1_routearrlinear_noslotlex_focus1_h095_r0_probe_20260709/gurobi_summary.json` |
| M4 | `middle_bomseq_m4_seed42_t900_g01_hist_sku22_16x5_bq33_qty34_stack3_copy1_support20_stationtop1_noslotlex_focus1_h005_r0_probe_20260710` | 1098 | 1088.113 | 0.932% | 567.1 | OPTIMAL | 24312 / 67269 | `result/middle_bomseq_m4_seed42_t900_g01_hist_sku22_16x5_bq33_qty34_stack3_copy1_support20_stationtop1_noslotlex_focus1_h005_r0_probe_20260710/gurobi_summary.json` |
| M5 | `middle_bomseq_m5_seed42_t900_g01_sku10_qty34_bq33_stationtop1_noslotlex_focus1_h005_r0_run_20260710` | 863 | 855.085 | 0.950% | 663.7 | OPTIMAL | 14377 / 39421 | `result/middle_bomseq_m5_seed42_t900_g01_sku10_qty34_bq33_stationtop1_noslotlex_focus1_h005_r0_run_20260710/gurobi_summary.json` |
| M6 | `middle_bomseq_m6_seed42_t1500_g01_sku14_qty34_bq33_stationtop1_noslotlex_focus1_h005_r0_run_20260710` | 1198 | 1188.12 | 0.884% | 582.7 | OPTIMAL | 56017 / 158503 | `result/middle_bomseq_m6_seed42_t1500_g01_sku14_qty34_bq33_stationtop1_noslotlex_focus1_h005_r0_run_20260710/gurobi_summary.json` |

M1-M6 结论：

- M1 因 360s time limit 停止，但 gap 0.541%，可作为当前链条起点证据。
- M2-M6 均为 OPTIMAL，gap 均小于 1%。
- Cmax 序列为 `582 -> 805 -> 830 -> 1098 -> 863 -> 1198`，其中 M5 低于 M4；如果论文链条要求严格 Cmax 单调，M5 需要单独解释为资源/结构变化口径，不能简单说 M1-M6 全部单调。
- M5 runtime 高于 M6 runtime，运行时间也不是严格单调；迁移时应保留这点风险说明。

#### 最新 M7 可收口径

接受结果目录：

```text
result/middle_bomseq_m7_seed42_t1500_g01_sku18_qty34_bq33_r5s3_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260710
```

关键结果：

| 指标 | 值 |
|---|---:|
| scale | M7 |
| seed | 42 |
| robots / stations | 5 / 3 |
| subtask / task | 24 / 25 |
| Cmax | 1538 |
| objective | 1538.590 |
| best bound | 1524.120 |
| gap | 0.9405% |
| gurobi runtime | 1314.21s |
| total runtime | 1335.33s |
| status | OPTIMAL |
| route arc | 43714 |
| warm_start_missing_arc_count | 0 |
| verification | `tra_makespan_verification.json` = PASS |

M7 当前有效参数：

- `time_limit=1500`
- `mip_gap=0.01`
- `candidate_stack_topk=999`
- `max_candidate_stacks_per_order=0`
- `candidate_station_topk_per_stack=1`
- `route_pickup_neighbor_limit=5`
- `enable_route_time_window_arc_prune=true`
- `enable_route_load_interval_arc_prune=true`
- `disable_slot_lex_symmetry=true`
- `gurobi_mip_focus=1`
- `gurobi_heuristics=0.05`
- `warm_start=true`
- `warm_start_use_sp4=true`
- `enable_warm_start_route_repair=true`
- 从目录名推断的问题参数：`sku18`、`exact qty U(3,4)`、`batch qty [3,3]`、`r5s3`

建议 Codex 复现命令模板：

```bash
/usr/local/bin/python3 experiments/run_global_xyzu.py \
  --scale M7 \
  --seed 42 \
  --time-limit 1500 \
  --mip-gap 0.01 \
  --runtime-config-json <需要固化的 M7 r5s3 runtime config> \
  --candidate-stack-topk 999 \
  --max-candidate-stacks-per-order 0 \
  --candidate-station-topk-per-stack 1 \
  --route-pickup-neighbor-limit 5 \
  --enable-route-time-window-arc-prune \
  --disable-resource-lex-symmetry \
  --disable-slot-lex-symmetry \
  --gurobi-mip-focus 1 \
  --gurobi-heuristics 0.05 \
  --output-root result/middle_bomseq_m7_seed42_t1500_g01_sku18_qty34_bq33_r5s3_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260710
```

迁移前待固化：

- 把 M7 r5s3 的问题生成配置写入一个明确文件，例如：
  - `experiments/configs/middle_m7_r5s3_accepted_20260710.json`
  - 或合并回 `experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json`
- 该配置应至少覆盖：`resources=[5,3,192]`、`exact_order_sku_counts=[18]*8`、`exact_order_sku_quantity_range=[3,4]`、`bom_batch_quantity_range=[3,3]`、`target_stack_count=32`、`middle_stack_shape` 保持 M7 现行布局。

#### M8 / M9 最新探测状态

M8 最新探测目录：

```text
result/middle_bomseq_m8_seed42_t2500_g01_sku22_qty34_bq33_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260711
```

结果：

- status=`TIME_VERIFY_MISMATCH`
- true/global makespan=`1598`
- model Cmax=`1605`
- gap=`12.9059%`
- runtime=`2500s` 级别
- route arc=`128405`
- warm_start_missing_arc_count=`0`

结论：M8 当前不能收。Cmax 高于 M7，规模方向是对的，但 gap 和 time verification mismatch 不满足论文安全口径。下一步需要优先解决 mismatch / 下界问题，而不是直接推进 M9。

M9 最新探测目录：

```text
result/middle_bomseq_m9_seed42_t3000_g01_sku10_qty34_bq33_stationtop1_noslotlex_focus1_h005_r0_run_20260711
result/middle_bomseq_m9_seed42_t3000_g01_sku10_qty34_bq33_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260711
```

结果摘要：

| 目录口径 | Cmax | gap | runtime | status | 结论 |
|---|---:|---:|---:|---|---|
| no-slot-lex | 806 | 1.1748% | 3000s | TIME_LIMIT | gap 接近但 Cmax 明显低于 M7/M8，不适合作为尾部链条 |
| twprune + route5 | 839 | 5.0963% | 3000s 级别 | TIME_LIMIT | gap 高，且 Cmax 仍偏低 |

结论：M9 当前只是探测证据，不能收。当前更合理的下一步是先把 M8 调到 gap < 1% 且 verification 通过，再重新设计 M9，避免出现更大 case 反而 Cmax/runtime 更弱。

## 3. 参数探测经验

已验证经验：

- 固定 seed=42 是当前论文链条的默认约束；不要为了好看随意换 seed。
- 地图/layout 是论文敏感配置；优先调 demand、batch、hit stacks、station/robot、route prune，不优先换地图。
- STACK 系列可以接受更强候选剪枝，因为目标是 runtime calibration；M-suite 更强调论文防御性，候选集默认应更保守。
- `candidate_stack_topk=999` + `max_candidate_stacks_per_order=0` 是中规模保守口径，避免候选截断改变问题本质。
- `route_pickup_neighbor_limit=5` + `enable_route_time_window_arc_prune=true` 在最新 M7 上有效，且 warm-start 缺边为 0。
- M7 从 4 station 调到 3 station 后，Cmax 和 runtime 进入合理链条，修复了更大算例反而过轻的问题。
- M8 不能只看 Cmax 高于 M7；必须同时看 gap、verification、warm-start 兼容性。

需要避免：

- 不要把 `TIME_VERIFY_MISMATCH` 当可收结果。
- 不要只用 `global_makespan` 报告；要同时检查 `model_cmax`、`true_global_makespan`、`tra_makespan_verification.json`。
- 不要混淆 STACK 的 `disable warm start` 口径和 M-suite 的 `warm start enabled` 口径。
- 不要把旧 `docs/medium_scale_results_20260616.md` 的 TRA-Fast 表当成当前 Gurobi calibration 终稿。

## 4. Codex 迁移最小文件包

建议最小迁移集：

```text
Gurobi/
problemDto/
entity/
config/
experiments/
tests/
test_*.py
environment.yml
docs/codex_migration_summary_20260711.md
docs/medium_scale_results_20260616.md
docs/middle_m1_m9_parameters_for_lark.xml
docs/stacks_s1_s9_gurobi_results_20260621.md
experiments/configs/stacked_single_block_runtime_configs.json
experiments/configs/stacked_single_block_runtime_report.md
experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json
```

建议同步的结果证据：

```text
result/stacked_single_block_runtime_s1_range_1_5_batch_1_3_station2_pruned_r0
result/stacked_single_block_runtime_s2_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s3_range_1_5_batch_1_3_station1_no_path_prune
result/stacked_single_block_runtime_s4_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s5_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s6_range_1_5_batch_1_3_station2_pruned_r0
result/stacked_single_block_runtime_s7_baseline_gurobi_log_t220
result/stacked_single_block_runtime_s8_range_1_5_batch_1_3_station2_pruned_r5
result/stacked_single_block_runtime_s9_seed42_qty_1_5_batch_3_5_stack211111_cand7_station2_r5_t260
result/middle_bomseq_m7_seed42_t1500_g01_sku18_qty34_bq33_r5s3_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260710
result/middle_bomseq_m8_seed42_t2500_g01_sku22_qty34_bq33_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260711
result/middle_bomseq_m9_seed42_t3000_g01_sku10_qty34_bq33_stationtop1_noslotlex_focus1_h005_r0_run_20260711
result/middle_bomseq_m9_seed42_t3000_g01_sku10_qty34_bq33_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260711
```

可不迁移或低优先：

- `diagrams/`：只影响展示。
- 大量 `result/task*`、`result/probe*`、`result/audit*`：历史探测很多，除非要复盘具体失败原因，否则不必全量迁移。
- `test_gurobi_cut/results*`：早期 Gurobi cut 探测证据，非当前主线。

## 5. Codex 接手顺序

建议按以下顺序跑：

1. 环境验证：

```bash
python - <<'PY'
import gurobipy
print("gurobi", gurobipy.gurobi.version())
PY
```

2. 单元/轻量回归：

```bash
python -m pytest tests test_global_xyzu_solver.py -q
```

3. STACK dry-run：

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py \
  --scales STACK-S9 \
  --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json \
  --seed 42 \
  --time-limit 1 \
  --mip-gap 0.01 \
  --candidate-stack-topk 7 \
  --candidate-station-topk-per-stack 2 \
  --route-pickup-neighbor-limit 5 \
  --disable-warm-start \
  --disable-warm-start-sp4 \
  --dry-run \
  --output-dir result/codex_s9_dryrun
```

4. 复现 STACK-S9 或任一 accepted STACK case。

5. 固化并复现 M7 r5s3。

6. 在 M7 可复现后，再处理 M8 的 `TIME_VERIFY_MISMATCH` 和 gap。

## 6. 当前风险与下一步

风险：

- M7 最新可收结果没有完全固化到主 `experiments/configs`，Codex 迁移后可能无法一键复现。
- M8 结果不合格，不能作为论文链条正式项。
- M9 当前设计过轻，Cmax 低于 M7/M8，不符合尾部算例直觉。
- 结果目录很多，Codex 侧若全量复制会浪费空间并混淆口径。

下一步建议：

1. 先新增一个明确的 M7 accepted runtime config 文件。
2. 用该文件在本机或 Codex 复现 M7，确认 Cmax/gap/runtime/verification 一致。
3. 为 M8 保留同一 `r5s3/twprune/route5` 思路，但优先修复 `TIME_VERIFY_MISMATCH` 与 gap。
4. M8 收敛后再重设 M9，避免使用当前 `sku10` 导致尾部 Cmax 偏低。
