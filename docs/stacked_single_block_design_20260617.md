# Stacked Single-Block Small Instance Design

This note records the first defensible stacked-warehouse small instance for exact Gurobi runs without MIP-start injection.

## Instance

Config file: `experiments/configs/stacked_single_block_runtime_configs.json`

Case: `STACK-S1`

| Item | Setting |
|---|---:|
| Map blocks | 1 x 1 |
| Physical grid | 9 x 11 nodes |
| Active stacks | 8 |
| Stack max height | 8 totes |
| Totes | 44 |
| Average stack height | 5.5 totes/stack |
| Stack height histogram | 4: 1 stack; 5: 4 stacks; 6: 2 stacks; 8: 1 stack |
| Robots | 2 |
| Stations | 2 |
| SKU universe | 18 |
| BOM count | 2 |
| SKU types per BOM | 7 |
| Part quantity per SKU in BOM | U(5,10) |
| Batch quantity | 5 x U(1,5) |
| Gurobi target | 300 s, MIPGap 0.01 |
| Warm start | disabled |

The physical block follows the current `WarehouseMap` layout: one cross aisle before and after the rack rows, vertical aisle columns between paired rack columns, and workstation nodes in the top row. Empty stack positions remain physically present in the single block, but only 12 stacks are activated for inventory so that the average tote height is realistic for a stacked system.

Layout image:

`result/stacked_single_block_design_20260617_batch/STACK-S1_layout.png`

## Validation

No-warm Gurobi command:

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' `
  --scales STACK-S1 `
  --seed 42 `
  --time-limit 300 `
  --mip-gap 0.01 `
  --candidate-stack-topk 3 `
  --route-pickup-neighbor-limit 5 `
  --candidate-station-topk-per-stack 1 `
  --disable-warm-start `
  --disable-order-time-windows `
  --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' `
  --big-m-time 10000 `
  --route-big-m-time 10000 `
  --output-dir 'result/stacked_single_block_design_20260617_batch/gurobi_no_warm_bigm10000_final'
```

Result:

| Status | Cmax | Best bound | Gap | Runtime |
|---|---:|---:|---:|---:|
| OPTIMAL | 2511.000000 | 2488.020000 | 0.009190 | 43.817588 s |

Model-size probe:

| Variables | Route arcs | passX vars | Compile time |
|---:|---:|---:|---:|
| 2967 | 1360 | 120 | 0.53 s |

The final solution genuinely splits each BOM:

| Order | Subtasks | SKU split |
|---:|---:|---|
| 0 | 2 | `[3, 6, 14]` and `[7, 10, 15, 17]` |
| 1 | 2 | `[4, 5, 9, 12]` and `[8, 11, 13]` |

For this batch instance, SKU quantities are not expanded into separate decision units. Each `(BOM, SKU)` remains one work unit; the quantity map only multiplies station picking time. Totes are treated as carrying an unlimited amount of any SKU they contain.

## BOM Splitting and Shared SKU Notes

Current `global_xyzu` preparation expands demand by `(order_id, sku_id)`, not by every physical part unit. The repeated quantity is stored as `demand_qty` and affects coverage/picking duration, while subtask slot count is mainly driven by distinct SKU count and robot capacity. This is the right direction for avoiding quantity-driven variable explosion.

The expensive dimension is sub-BOM slots. If a BOM is split into many slots, variables grow roughly with:

- `x`: demand work units x candidate slots.
- `y`: candidate slots x station/rank choices.
- `z/sort/hit/noise`: candidate slots x candidate stacks/totes/stack intervals.
- `u route`: pickup-delivery tasks and route arcs grow fastest, often near quadratic in route nodes before pruning.

For exact small cases that must demonstrate splitting, keep each BOM above the effective transport capacity. In this code the slot lower bound uses `ROBOT_CAPACITY - 2`; with `ROBOT_CAPACITY=8`, a 6-SKU BOM can stay as one subtask, while a 7-SKU BOM forces at least two subtask slots. For thesis-scale manufacturing BOMs with 50-70 SKU types, do not model every split as an unconstrained slot; use capacity-derived slot bounds and candidate-stack pruning.

Shared SKU across different BOMs needs careful treatment. A physical SKU should be a shared inventory item, but each BOM still has its own demand quantity and completion requirement. The clean formulation is:

- Keep demand units keyed by `(bom_id, sku_id)`.
- Keep inventory keyed by physical `sku_id`.
- Allow different BOMs to use the same physical SKU/tote over time, with route/time constraints deciding repeated visits.
- Do not globally consume a tote for only one BOM unless the model explicitly represents depleted quantity.

In a quick shared-SKU stress setup, fully shared BOMs caused infeasibility in the current integrated model even after widening candidates. Treat that as a modeling audit item before using common-part BOMs in formal experiments.
