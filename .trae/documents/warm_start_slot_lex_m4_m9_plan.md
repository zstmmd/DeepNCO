# warm start slot lex 与 M4-M9 调参计划

## Summary

目标是继续修复 Global XYZU 的 warm start 注入，使启用 `SlotLoadLex` / `SlotStationLex` 时，warm start 的 slot 分配和站台 rank start 值本身满足这两类约束；随后按 m4 -> m9 逐轮调参运行，并把唯一正式结果表写入 `experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json`。

执行阶段只接受 M4-M9 的最终正式结果，不再把中间探针/废弃版本写进目标配置。M1-M3 保留既有正式结果：

| case | 上界 | 下界 | cmax | gap | runtime |
|---|---:|---:|---:|---:|---:|
| M1 | 582.208 | 579.06 | 582 | 0.54% | 360.180s |
| M2 | 805.27 | 798.09 | 805 | 0.89% | 520.171s |
| M3 | 827.376 | 822.09 | 827 | 0.64% | 700.262s |

## Resume State

2026-06-28 恢复上下文后做过只读确认：

- 计划文件已存在于 `.trae/documents/warm_start_slot_lex_m4_m9_plan.md`，执行阶段继续沿用本文件。
- `Gurobi/global_xyzu.py` 已经插入 `_canonical_warm_slot_profiles`、`_lex_aware_station_rank_rows`、`_validate_slot_lex_starts` 等 helper；执行阶段不要重复插入，重点是接入调用点。
- `_apply_warm_start` 目前仍按 `SubTask.id` 映射 slot，并在后段按 route arrival 重排 station rank；这是待修复的主路径。
- `_estimate_warm_model_cmax_for_route_prune` 目前也仍按 `SubTask.id` 映射 slot，repair 分支内 station rank 也按 arrival 排；需要与正式 warm start 逻辑保持一致。
- `GlobalXYZUSolver` diagnostics 已有 `model_constr_count_total` 和 `model_constr_count_by_type`，但 `experiments/probe_gurobi_model_size.py` 还没有把约束数字段写入 CSV/console。
- 目标配置文件顶层结构为 `description/reference/common_solver/configs/case_runs/results/acceptance`；M4 已是每 BOM SKU 22，M5-M9 仍需改成用户指定的 10/14/18/22/10。

## Current State Analysis

- `Gurobi/global_xyzu.py` 已有 slot lex 约束：
  - `SlotLoadLex_*`：同一 BOM 的 slot 按 local index 要求 `sku_use` 数量非递增。
  - `SlotStationLex_*`：相邻 slot 在 load 相等时要求 `(station_id, rank)` 字典序非递减。
- 当前 `_apply_warm_start` 先按 `SubTask.id` 把 warm subtask 填进 slot，再设置 `y` 的 station/rank start；这与 `SlotLoadLex` 需要的 load 排序没有绑定。
- 当前 `_apply_warm_start` 后段还会按 route arrival 重排 station rank。这个排序有利于时间起点，但在同一订单、同 load、同 station 的 slot 上可能反向，导致 `SlotStationLex` 的 MIP Start 不可行。
- `_estimate_warm_model_cmax_for_route_prune` 也按旧顺序构造 warm slot/rank 估计，执行阶段需要同步修正，避免 route prune 上界与真实 warm start 逻辑不一致。
- `experiments/probe_gurobi_model_size.py` 能 compile 模型并输出变量数，但当前 CSV/console 对总约束数支持不完整；用户要求先看变量数和约束数再决定是否完整运行，需要补齐。
- 目标配置 `experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json` 是有效 JSON，已有 M1-M3 正式结果和历史 probe；M4-M9 当前还没有正式结果。
- `tmp/middle_stack_bomseq_runtime_configs.json` 含大量 2026-06-27 的历史/废弃 probe，其中 M5-M8 用过修改 batch 的 `bq1214` 配置，已经被历史说明标记 invalidated。本次不会复制这些配置或结果。
- 当前目标配置里 M5-M9 的每 BOM SKU 数与本次要求不一致，需要执行阶段修正：
  - M4 = 22
  - M5 = 10
  - M6 = 14
  - M7 = 18
  - M8 = 22
  - M9 = 10

## Assumptions & Decisions

- 固定不改：BOM 数、总 SKU 数、tote、stack、robot、station、batch 取当前目标配置中的值。
- 每 BOM SKU 数按用户给出的 M4-M9 序列作为硬约束，写到 `bom_complexity[0]` 与 `exact_order_sku_counts`；不接受低于或高于该指定值的正式配置。
- 允许调参的范围：
  - `bom_colocated_stack_counts`
  - `bom_colocated_support_multiplier`
  - `bom_colocated_chunked_by_stack`
  - `middle_stack_shape` / 地图 SKU 排布方向
  - solver candidate 参数：`candidate_stack_topk`、`max_candidate_stacks_per_order`、`candidate_station_topk_per_stack`
  - `subtask数` 通过允许的 inventory/layout 分布间接控制
- 不使用 `--disable-slot-lex-symmetry` 作为正式跑法；M4-M9 正式命令必须启用 slot lex。
- `route_pickup_neighbor_limit` 初始保持 0；只有完整模型尺寸明显过大且用户约束仍允许时，才作为探针参数，不直接写正式结果。
- 验收阈值：gap 最好 <= 1.00%，必须 < 1.15%；`coverage_ok=true`、`makespan_consistent=true`、`has_unreasonable_solution=false`。
- 运行时间目标：要求整体递增，接近可接受。初始正式时间窗按 M4=780s、M5<=1200s、M6<=1500s、M7<=1800s、M8<=2100s、M9=2500-3600s 处理。
- 当前工作区已有大量未提交改动；执行阶段只做本计划涉及文件的最小补丁，不回滚其它文件。

## Proposed Changes

### 1. `Gurobi/global_xyzu.py`

修复 warm start start 值生成逻辑。

具体做法：

1. 复用已插入的 warm subtask profile/canonical helper，用于同一订单内排序 warm row：
   - 计算与 MIP Start 中 `sku_use` 一致的 `start_load`，优先统计能被 warm hit tote 覆盖的 unique SKU；无法覆盖的 SKU 不计入 start load，保持与现有 `warm_start_uncovered_sku_start_skipped_count` 逻辑一致。
   - 记录 `station_id`、原始 `station_sequence_rank`、warm 到站时间、原始 subtask id。
   - 排序 key：`(-start_load, station_id, original_rank, warm_arrival, subtask_id)`。
   - 把排序后的 rows 映射到 `slot_ids_by_order[order_id]`，替代当前按 `SubTask.id` 的直接映射。
2. 复用已插入的 slot lex start 校验 helper：
   - 对每个订单读取 active slot start、`sku_use.Start`、`y.Start`。
   - 验证相邻 slot 的 load 非递增。
   - 当相邻 load 相等时，验证 station/rank code 非递减。
   - 输出 diagnostics：`warm_start_slot_lex_checked`、`warm_start_slot_load_lex_violation_count`、`warm_start_slot_station_lex_violation_count`、样例 rows。
3. 替换现有 “按 route arrival 重排 station rank” 的后段逻辑：
   - 保留 route arrival 作为时间重建输入。
   - station rank Start 改为 lex-aware 排序：每个 station 内按 `(order_id, order_local_slot_index, start_load, slot_id)` 为主、arrival 为次要 tie-break，确保同一订单同 load 且同 station 的 slot rank 不反向。
   - 排完 rank 后重新写 `y.Start`，再调用 `_rebuild_warm_slot_continuous_start` 生成连续时间 start。
   - rank 改写后再次跑 slot lex start 校验；若仍有 violation，不把 `warm_start_mip_start_ready` 标记为 true，并把 violation 诊断写出。
4. 同步修正 `_estimate_warm_model_cmax_for_route_prune`：
   - 使用同一套 canonical slot order 与 lex-aware rank row 逻辑。
   - 避免 route prune bound 估计使用与正式 MIP Start 不同的 slot/rank 顺序。

### 2. `test_global_xyzu_solver.py`

补充 targeted 单测。

新增或扩展测试：

1. `test_warm_start_slot_lex_order_canonicalizes_load_and_station_rank`
   - 构造/操纵一个小 warm start，使 warm rows 的原始 subtask id 顺序不满足 load/rank lex。
   - 调 `_apply_warm_start` 后检查 diagnostics 中 slot lex violation 计数为 0。
   - 直接读取 `sku_use.Start` 与 `y.Start`，验证 `SlotLoadLex` / `SlotStationLex` 的 start 值语义。
2. 扩展 `test_apply_warm_start_rebuilds_continuous_starts`
   - 增加断言：`warm_start_slot_lex_checked=true`，load/station violation 均为 0。

### 3. `experiments/probe_gurobi_model_size.py`

补齐模型尺寸探针。

具体做法：

1. 在 `_row_from_diag` 暴露已有 solver diagnostics 中的约束字段：
   - `model_constr_count_total`
   - `model_linear_constr_count_total`
   - `model_general_constr_count_total`
   - `slot_load_lex_count`
   - `slot_station_lex_count`
2. 新增 `run_like_slotlex` probe config，尽量贴近 `run_global_xyzu.py` 的正式命令：
   - warm start enabled
   - slot lex enabled
   - candidate stack unrestricted by默认 `999/0`
   - station topk 可通过参数调整，默认 999
   - pickup KNN disabled
   - safe route arc/load interval prune enabled
3. console 输出同时打印 vars / constraints / slot lex count，方便决定是否进入完整运行。

### 4. `experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json`

执行阶段更新目标配置，不写 tmp 文件作为正式记录。

配置修正：

- 保留 M1-M3 正式配置与正式结果。
- M4-M9 保留当前 BOM 数、总 SKU、tote、stack、robot、station、batch。
- 更新 M4-M9 的每 BOM SKU 数：
  - M4：当前已正确，保持 `bom_complexity[0]=22`，`exact_order_sku_counts=[22,22,22,22,22,22]`
  - M5：`bom_complexity[0]=10`，`exact_order_sku_counts=[10,10,10,10,10,10,10,10]`
  - M6：`bom_complexity[0]=14`，`exact_order_sku_counts=[14,14,14,14,14,14,14,14]`
  - M7：`bom_complexity[0]=18`，`exact_order_sku_counts=[18,18,18,18,18,18,18,18]`
  - M8：`bom_complexity[0]=22`，`exact_order_sku_counts=[22,22,22,22,22,22,22,22]`
  - M9：`bom_complexity[0]=10`，`exact_order_sku_counts=[10,10,10,10,10,10,10,10,10,10]`
- `case_runs.M4` 初始设置为 780s / gap 1% / slot lex enabled 的命令。
- M5-M9 的 `case_runs` 只写最后接受命令，不保留探针命令。
- 增加或更新 `result_table`，字段严格包含：
  `case`, `BOM数`, `每BOM SKU数`, `总SKU数`, `tote`, `stack`, `robot`, `station`, `总需求量`, `变量数`, `约束数`, `命中stack数`, `subtask数`, `flip的tote数`, `sort的tote数`, `上界`, `下界`, `cmax`, `gap`, `runtime`, `command`。

## Execution Plan

### Step A: 修 warm start 并验证

1. 改 `Gurobi/global_xyzu.py` 与 `test_global_xyzu_solver.py`。
2. 运行 targeted 单测：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m unittest test_global_xyzu_solver.GlobalXYZUSolverTests.test_apply_warm_start_rebuilds_continuous_starts test_global_xyzu_solver.GlobalXYZUSolverTests.test_warm_start_slot_lex_order_canonicalizes_load_and_station_rank
```

3. 运行相关 smoke：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m unittest test_global_xyzu_solver.py
```

### Step B: 修 M4-M9 配置基础字段

1. 更新目标配置的 M4-M9 per-BOM SKU 数。
2. 保持 batch 与资源规模不变。
3. 用 JSON 校验：

```bash
jq '.configs | {M4,M5,M6,M7,M8,M9}' experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json
```

### Step C: 先看变量数和约束数

执行模型尺寸探针：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 experiments/probe_gurobi_model_size.py --cases M4,M5,M6,M7,M8,M9 --configs run_like_slotlex --seed 42 --time-limit 1 --mip-gap 0.01 --runtime-config-json experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json --output-dir result/middle_bomseq_model_size_probe_20260628
```

决策阈值：

- `vars <= 180000` 且 `constraints <= 550000`：允许直接跑正式时间窗。
- `180000 < vars <= 260000` 或 `550000 < constraints <= 800000`：先跑 120s 短探针，看是否有 incumbent 和有限 bound，再决定正式运行。
- `vars > 260000` 或 `constraints > 800000`：先缩规模，不跑完整窗口。

缩规模优先级：

1. 降 `candidate_station_topk_per_stack`：999 -> 2 -> 1。
2. 降 `bom_colocated_stack_counts` 或 support multiplier，减少命中 stack 数。
3. 调整 `middle_stack_shape` 方向，控制地图 SKU 分布与候选站台距离。
4. 最后才调整 candidate stack topk / max candidate stacks；正式结果仍优先保持 pickup KNN 为 0。

### Step D: 逐 case 完整运行与调参

正式命令模板：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 experiments/run_global_xyzu.py --scale <CASE> --seed 42 --time-limit <TIME_LIMIT> --mip-gap 0.01 --runtime-config-json experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json --candidate-stack-topk <STACK_TOPK> --max-candidate-stacks-per-order <MAX_STACKS> --candidate-station-topk-per-stack <STATION_TOPK> --route-pickup-neighbor-limit 0 --disable-resource-lex-symmetry --gurobi-mip-focus <FOCUS> --gurobi-heuristics <HEURISTICS> --output-root result/middle_bomseq_<case>_seed42_t<limit>_g01_slotlex_<tag>_20260628
```

初始时间窗：

- M4：780s，先跑，目标 gap <= 1%，必须 < 1.15%，runtime 约 780s 且不超过 800s。
- M5：1200s 以内。
- M6：1500s 以内。
- M7：1800s 以内。
- M8：2100s 以内。
- M9：先 3000s；如果 gap 未到 1.15%，允许提高到 3600s 内。

每个 case 的调参循环：

1. 先 probe 模型尺寸。
2. 如果尺寸可接受，跑正式窗口。
3. 如果无 incumbent 或 gap 太差：
   - 提高 `MIPFocus` 到 3、降低 `Heuristics` 到 0.02 试证明下界。
   - 若 incumbent 不好，改用 `MIPFocus=1`、`Heuristics=0.3/0.5`。
   - 必要时打开 `--enable-route-arrival-slot-linear` 做对照探针。
4. 如果 runtime 不递增：
   - 过快：优先增加命中 stack 数或 station topk；不改 BOM 数、batch、总 SKU、tote/stack/robot/station。
   - 过慢：优先减少命中 stack 数或 station topk；不改固定字段。
5. 只有当前 case 达到 gap/runtime/审计要求后，才进入下一个 case。

### Step E: 结果抽取与写表

每次正式运行后从 `<output-root>/gurobi_summary.json` 与 `<output-root>/gurobi_solution_export/best_solution_audit.json` 读取：

- `status`
- `objective` 作为上界
- `diagnostics.model_best_bound` 作为下界
- `diagnostics.model_gap`
- `diagnostics.gurobi_runtime_sec`
- `diagnostics.model_var_count_total`
- `diagnostics.model_constr_count_total`
- `subtask_count`
- `global_makespan` / `true_global_makespan`
- `orders[].total_qty` 汇总为总需求量
- `tasks[]` 中按 `mode=FLIP/SORT` 汇总 `target_tote_ids` 数，得到 `flip的tote数` / `sort的tote数`
- `diagnostics.candidate_stack_count_by_order` 或最终 task stack 去重数，记录为 `命中stack数`

写入目标配置：

- `case_runs.M4` ... `case_runs.M9`：只保留接受的正式命令。
- `results.M4` ... `results.M9`：写正式结果摘要。
- `result_table.rows`：写完整 M1-M9 表格，M4-M9 含运行命令。

## Verification Steps

执行阶段完成后必须验证：

1. JSON 有效：

```bash
jq empty experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json
```

2. M4-M9 每 BOM SKU 数正确：

```bash
jq '.configs | {M4:.M4.exact_order_sku_counts,M5:.M5.exact_order_sku_counts,M6:.M6.exact_order_sku_counts,M7:.M7.exact_order_sku_counts,M8:.M8.exact_order_sku_counts,M9:.M9.exact_order_sku_counts}' experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json
```

3. 每个接受结果：
   - `model_gap < 0.0115`
   - `coverage_ok=true`
   - `makespan_consistent=true`
   - `has_unreasonable_solution=false`
   - `warm_start_slot_load_lex_violation_count=0`
   - `warm_start_slot_station_lex_violation_count=0`
4. runtime 序列从 M1 到 M9 基本递增；M4-M9 满足目标窗口或记录“接近”原因。
5. `result_table.rows` 字段完整，且 M4-M9 每行都有 `command`。
