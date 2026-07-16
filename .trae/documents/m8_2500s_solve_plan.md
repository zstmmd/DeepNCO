# M8 2500s 求解计划

## Summary

目标是在不修改 M8 实例口径的前提下，使当前 M8 正式算例在 2500s 内达到：

- `model_gap <= 0.01`
- `warm_start_mip_start_ready=true`
- `warm_start_missing_arc_count=0`
- `time_verify_mismatch=false`
- 不接受 `WARM_START_FALLBACK`
- 不修改 BOM、SKU、tote、stack-grid、map、robot、station 等论文敏感配置

当前证据显示，M8 不是 warm start 不可行，也不是单纯 route Big-M 数值过大；最大问题是 integrated U 模型在 root node 阶段被十万级 general indicator constraints 拖住，导致 300s/1800s 都无法进入有效 branch-and-bound，root bound 长期停在 `1398.91`。

计划主线：**保持 route arc policy 不变，优先把 U 层 indicator constraints 替换为等价 linear Big-M constraints，减少 Gurobi root 阶段 general constraint 处理负担。**

## Current State Analysis

### 当前 M8 配置

配置文件：

- `experiments/configs/m8_map5x10_tote300_sku340_coloc6_chunktrue_20260712.json`

关键口径：

- map: `5x10`
- stack: `50`
- tote: `300`
- robot: `5`
- station: `4`
- order: `8`
- SKU: `340`
- 每单 SKU: `22`
- 每单 colocated stack count: `6`
- `storage_gap_rows=4`
- `bom_colocated_chunked_by_stack=true`

密集存储口径满足：`300 / 50 = 6 > 5`。

### M7 vs M8 差距

M7 基线：

- `result/middle_bomseq_m7_seed42_t1500_g01_sku18_qty34_bq33_r5s3_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260710/gurobi_summary.json`

M8 基线：

- `result/m8_map5x10_tote300_sku340_coloc6_chunktrue_route5_allcuts_t300_focus1_h005_20260712/gurobi_summary.json`

| 指标 | M7 | M8 | 变化 |
| --- | ---: | ---: | ---: |
| `slot_count` | 24 | 32 | 1.33x |
| `work_unit_count` | 144 | 176 | 1.22x |
| `u_candidate_task_count` | 153 | 268 | 1.75x |
| `u_arc_count` | 43,714 | 117,412 | 2.69x |
| `model_var_count_total` | 53,965 | 134,463 | 2.49x |
| `model_constr_count_total` | 149,129 | 384,939 | 2.58x |
| `model_general_constr_count_total` | 134,783 | 358,699 | 2.66x |
| `model_best_bound` | 1524.12 | 1398.91 | 下降 |
| `model_gap` | 0.94% | 15.18% | 失控 |
| `model_node_count` | 6971 | 1 | 卡 root |

### 已排除方向

以下方向已验证不能作为主路径：

1. `warm route time upper bound`
   - 300s: `BestBound=1398.91`
   - 结论：只收紧时间上界不能抬 bound。

2. 额外添加 per-arc route time Big-M linear cut
   - 300s: `BestBound=1398.91`
   - 问题：只是增加冗余线性约束，没有移除原 indicator burden。

3. delivery→pickup neighbor prune
   - 能降低 arc 数，但出现 `warm_start_missing_arc_count=1` 或 fallback。
   - 结论：不进入正式验收组合。

4. `SlotPairArrivalLB` / `SlotSkuArrivalLB` / `SkuReleaseWorkloadLB`
   - LP relaxation 仍为 `1398.91`
   - 紧约束仍是 `GlobalArrivalWorkloadLB` 与 `StationReleaseSuffixWorkloadLB_15_0`

5. `StationRankWorkloadLB`
   - LP relaxation 仍为 `1398.91`
   - 说明普通 rank workload prefix LB 不足以抬 root bound。

## Proposed Changes

### 1. 新增 route constraint mode

文件：

- `Gurobi/global_xyzu.py`
- `experiments/run_global_xyzu.py`

新增配置：

```python
route_constraint_mode: str = "indicator"
```

新增 CLI：

```bash
--route-constraint-mode indicator|linear
```

默认值必须是 `indicator`，保证历史实验默认行为不变。

M8 2500s 方案使用：

```bash
--route-constraint-mode linear
```

### 2. U 层 RouteTimeCont 从 indicator 替换为 linear Big-M

位置：

- `Gurobi/global_xyzu.py`
- U 层 `for i, j in route_arcs` route time continuity 约束块

当前语义：

```text
route_arc[i,j] = 1 =>
route_time[j] >= route_time[i] + service[i] + tau[i,j]
```

linear 模式改为：

```text
route_time[j] >= route_time[i] + service[i] + tau[i,j] - M_time[i,j] * (1 - route_arc[i,j])
```

`M_time[i,j]` 取值优先级：

1. `route_arc_time_m[(i,j)]`
2. `route_node_time_ub[i] + pickup_service_ub_by_node[i] + tau[i,j]`
3. `slot_time_ub`

要求：

- `route_constraint_mode="linear"` 时不再添加 `RouteTimeCont` general constraint。
- 增加 diagnostics：
  - `route_time_indicator_count`
  - `route_time_linear_count`
  - `route_constraint_mode`

### 3. U 层 RouteLoadLB 从 indicator 替换为 linear Big-M

当前语义：

```text
route_arc[i,j] = 1 =>
route_load[j] == route_load[i] + demand[j]
```

linear 模式改为：

```text
route_load[j] >= route_load[i] + demand[j] - M_load * (1 - route_arc[i,j])
route_load[j] <= route_load[i] + demand[j] + M_load * (1 - route_arc[i,j])
```

`M_load`：

```text
2 * robot_capacity
```

要求：

- `linear` 模式不再添加 `RouteLoadLB` general constraint。
- 增加 diagnostics：
  - `route_load_indicator_count`
  - `route_load_linear_count`

### 4. U 层 owner sync 从 indicator 替换为 linear Big-M

适用对象：

- `RouteOwnerSync`
- `RouteStartOwnerLink`
- `RouteEndOwnerLink`

当前语义：

```text
route_arc[i,j] = 1 => route_owner[i] == route_owner[j]
```

linear 模式改为：

```text
route_owner[j] - route_owner[i] <= M_owner * (1 - route_arc[i,j])
route_owner[i] - route_owner[j] <= M_owner * (1 - route_arc[i,j])
```

`M_owner`：

```text
max(1, len(robot_ids) - 1)
```

要求：

- `linear` 模式不再添加 owner sync general constraints。
- 增加 diagnostics：
  - `route_owner_indicator_count`
  - `route_owner_linear_count`

### 5. 正式组合保持 route arc policy 不变

正式组合不启用：

- `enable_route_delivery_pickup_neighbor_prune`

保持：

- `route_pickup_neighbor_limit=5`
- `candidate_station_topk_per_stack=1`
- `route_arc_prune=True`
- `enable_route_time_window_arc_prune=True`
- `enable_route_load_interval_arc_prune=True`

原因：

- route5 当前 `warm_start_missing_arc_count=0`
- delivery→pickup prune 已证明破坏 warm start 稳定性

### 6. 不把已证明无效的下界放入第一轮正式组合

第一轮 linear-U 探针不启用：

- `enable_slot_pair_arrival_lb`
- `enable_slot_sku_arrival_lb`
- `enable_sku_release_workload_lb`
- `enable_station_rank_workload_lb`

原因：

- 它们的 LP relaxation 仍为 `1398.91`
- 第一轮必须隔离验证 indicator replacement 的效果

## Implementation Steps

1. 在 `GlobalXYZUConfig` 增加 `route_constraint_mode: str = "indicator"`。
2. 在 `experiments/run_global_xyzu.py` 增加 `--route-constraint-mode` 参数并传入 config。
3. 在 `Gurobi/global_xyzu.py` 中 U 层 route arc loop 做 mode 分支：
   - `indicator`: 保持当前 `addGenConstrIndicator`
   - `linear`: 添加等价 linear Big-M，不添加对应 indicator
4. 对 route owner sync 的所有 indicator 分支做同样替换。
5. 增加诊断计数，确保 1s probe 能看到 general constraints 明显下降。
6. 保留当前默认行为，避免影响历史 M7/M8 baseline。

## Verification Steps

### 1. 静态检查

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m py_compile \
  Gurobi/global_xyzu.py \
  experiments/run_global_xyzu.py

python3 -m py_compile \
  Gurobi/global_xyzu.py \
  experiments/run_global_xyzu.py
```

### 2. 1s probe

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 experiments/run_global_xyzu.py \
  --scale M8 \
  --seed 42 \
  --time-limit 1 \
  --mip-gap 0.01 \
  --runtime-config-json experiments/configs/m8_map5x10_tote300_sku340_coloc6_chunktrue_20260712.json \
  --candidate-stack-topk 999 \
  --max-candidate-stacks-per-order 0 \
  --candidate-station-topk-per-stack 1 \
  --route-pickup-neighbor-limit 5 \
  --enable-route-time-window-arc-prune \
  --enable-slot-min-arrival-lb \
  --enable-route-incident-travel-lb \
  --enable-route-pair-service-travel-lb \
  --enable-route-finish-cmax-lb \
  --enable-route-arrival-slot-linear \
  --enable-uz-lb-cuts \
  --route-constraint-mode linear \
  --disable-resource-lex-symmetry \
  --disable-slot-lex-symmetry \
  --gurobi-mip-focus 1 \
  --gurobi-heuristics 0.05 \
  --quiet-gurobi \
  --output-root result/m8_map5x10_tote300_sku340_coloc6_chunktrue_route5_linearU_probe_t1_20260713
```

必须检查：

- `status != WARM_START_FALLBACK`
- `warm_start_mip_start_ready=true`
- `warm_start_missing_arc_count=0`
- `time_verify_mismatch=false`
- `route_constraint_mode=linear`
- `route_time_indicator_count=0`
- `route_load_indicator_count=0`
- `route_owner_indicator_count=0`
- `route_time_linear_count > 0`
- `route_load_linear_count > 0`
- `route_owner_linear_count > 0`
- `model_general_constr_count_total` 相对 baseline 明显下降

### 3. 300s probe

只有 1s probe 通过才跑：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 experiments/run_global_xyzu.py \
  --scale M8 \
  --seed 42 \
  --time-limit 300 \
  --mip-gap 0.01 \
  --runtime-config-json experiments/configs/m8_map5x10_tote300_sku340_coloc6_chunktrue_20260712.json \
  --candidate-stack-topk 999 \
  --max-candidate-stacks-per-order 0 \
  --candidate-station-topk-per-stack 1 \
  --route-pickup-neighbor-limit 5 \
  --enable-route-time-window-arc-prune \
  --enable-slot-min-arrival-lb \
  --enable-route-incident-travel-lb \
  --enable-route-pair-service-travel-lb \
  --enable-route-finish-cmax-lb \
  --enable-route-arrival-slot-linear \
  --enable-uz-lb-cuts \
  --route-constraint-mode linear \
  --disable-resource-lex-symmetry \
  --disable-slot-lex-symmetry \
  --gurobi-mip-focus 1 \
  --gurobi-heuristics 0.05 \
  --gurobi-start-node-limit 500 \
  --quiet-gurobi \
  --output-root result/m8_map5x10_tote300_sku340_coloc6_chunktrue_route5_linearU_t300_20260713
```

进入 2500s 的门槛：

- `model_node_count > 1`
- 或 `model_best_bound > 1450`
- 或 `model_gap <= 0.10`
- 同时：
  - `status != WARM_START_FALLBACK`
  - `warm_start_mip_start_ready=true`
  - `warm_start_missing_arc_count=0`
  - `time_verify_mismatch=false`

若 300s 仍为：

- `model_best_bound=1398.91`
- `model_node_count=1`

则不跑 2500s，进入第二轮：station clock indicator linearization。

### 4. 2500s formal

若 300s 通过门槛：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 experiments/run_global_xyzu.py \
  --scale M8 \
  --seed 42 \
  --time-limit 2500 \
  --mip-gap 0.01 \
  --runtime-config-json experiments/configs/m8_map5x10_tote300_sku340_coloc6_chunktrue_20260712.json \
  --candidate-stack-topk 999 \
  --max-candidate-stacks-per-order 0 \
  --candidate-station-topk-per-stack 1 \
  --route-pickup-neighbor-limit 5 \
  --enable-route-time-window-arc-prune \
  --enable-slot-min-arrival-lb \
  --enable-route-incident-travel-lb \
  --enable-route-pair-service-travel-lb \
  --enable-route-finish-cmax-lb \
  --enable-route-arrival-slot-linear \
  --enable-uz-lb-cuts \
  --route-constraint-mode linear \
  --disable-resource-lex-symmetry \
  --disable-slot-lex-symmetry \
  --gurobi-mip-focus 1 \
  --gurobi-heuristics 0.05 \
  --gurobi-start-node-limit 500 \
  --quiet-gurobi \
  --output-root result/m8_map5x10_tote300_sku340_coloc6_chunktrue_route5_linearU_t2500_20260713
```

正式通过条件：

- `model_gap <= 0.01`
- `time_verify_mismatch=false`
- `warm_start_mip_start_ready=true`
- `warm_start_missing_arc_count=0`
- `model_cmax == validated_global_makespan`
- `model_cmax > M7 model_cmax`

## Risks

- Linear Big-M 替换如果 M 值过小，会导致模型不可行或错剪可行域。
- Linear Big-M 替换如果 M 值过大，可能 LP 更松，但 root general constraints 会显著减少。
- 若 linear mode 出现异常更优 Cmax，优先怀疑约束缺失，不可直接收。
- 若 warm start 变为 infeasible，必须修 M 值或线性化语义，不可忽略。

## Rollback

- `route_constraint_mode` 默认保持 `indicator`。
- 若 linear mode 不通过，正式实验继续使用原 baseline，不污染 M8 结果链。
- 不启用 delivery→pickup prune 作为 fallback。
