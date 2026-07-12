# STACK-S1 到 STACK-S9 Gurobi Runtime 调参记录

## 目标

设计 `STACK-S1` 到 `STACK-S9` 单 block 密集堆垛式算例，使 Gurobi runtime 整体递增并控制在约 `20s-260s` 区间内。所有算例固定 `seed=42`，平均 tote/stack 大于 `5.5`。

本轮最终配置文件：

```text
experiments/configs/stacked_single_block_runtime_configs.json
```

保留的最终结果目录：

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

## 最终参数

| Case | resources `[robots, stations, totes]` | data `[BOM, total SKU]` | BOM complexity | exact_order_sku_counts | exact qty | batch qty | colocated stacks | support | copy | chunked | avg tote/stack |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|---:|
| STACK-S1 | `[2,2,48]` | `[2,18]` | `[7,1]` | `[7,7]` | `[1,5]` | `[1,3]` | `[3,2]` | 2.4 | 2 | false | 6.00 |
| STACK-S2 | `[2,2,48]` | `[2,32]` | `[10,1]` | `[10,10]` | `[1,5]` | `[1,3]` | `[3,3]` | 1.8 | 1 | true | 6.00 |
| STACK-S3 | `[2,2,52]` | `[2,46]` | `[14,1]` | `[14,14]` | `[1,5]` | `[1,3]` | `[3,3]` | 2.0 | 1 | false | 6.50 |
| STACK-S4 | `[2,2,56]` | `[2,60]` | `[18,1]` | `[18,18]` | `[1,5]` | `[1,3]` | `[3,3]` | 2.0 | 1 | true | 7.00 |
| STACK-S5 | `[3,2,60]` | `[4,74]` | `[7,1]` | `[7,7,7,7]` | `[1,5]` | `[1,3]` | `[2,2,2,2]` | 2.2 | 2 | true | 7.50 |
| STACK-S6 | `[3,2,60]` | `[4,88]` | `[10,1]` | `[10,10,10,10]` | `[1,5]` | `[1,3]` | `[2,2,2,2]` | 2.2 | 2 | true | 7.50 |
| STACK-S7 | `[3,2,62]` | `[4,102]` | `[15,1]` | `[15,15,15,15]` | `[1,5]` | `[1,3]` | `[2,2,2,2]` | 2.4 | 2 | true | 7.75 |
| STACK-S8 | `[3,2,64]` | `[4,116]` | `[18,1]` | `[18,18,18,18]` | `[1,5]` | `[1,3]` | `[2,2,2,2]` | 2.4 | 2 | true | 8.00 |
| STACK-S9 | `[3,2,64]` | `[6,130]` | `[7,1]` | `[7,7,7,7,7,7]` | `[1,5]` | `[3,5]` | `[2,1,1,1,1,1]` | 1.2 | 1 | false | 8.00 |

## 最终结果

| Case | seed | total_order_qty | subtask | task | hit stack | vars | route arc | Cmax | LB | gap | runtime | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| STACK-S1 | 42 | 58 | 4 | 8 | 5 | 1887 | 1200 | 107.0000 | 106.0512 | 0.009903 | 14.65s | OPTIMAL |
| STACK-S2 | 42 | 90 | 4 | 6 | 6 | 1905 | 1036 | 163.0000 | 163.0900 | 0.000000 | 15.78s | OPTIMAL |
| STACK-S3 | 42 | 111 | 6 | 9 | 6 | 2801 | 1336 | 239.0000 | 239.0479 | 0.000201 | 23.93s | OPTIMAL |
| STACK-S4 | 42 | 152 | 6 | 9 | 6 | 3587 | 1984 | 258.0000 | 258.1020 | 0.000000 | 50.33s | OPTIMAL |
| STACK-S5 | 42 | 169 | 8 | 9 | 7 | 2794 | 1317 | 269.0000 | 266.5424 | 0.009608 | 62.05s | OPTIMAL |
| STACK-S6 | 42 | 247 | 8 | 9 | 6 | 3762 | 2237 | 386.0000 | 383.5406 | 0.006747 | 76.16s | OPTIMAL |
| STACK-S7 | 42 | 338 | 12 | 12 | 5 | 2346 | 480 | 526.0000 | 520.0875 | 0.011504 | 220.03s | TIME_LIMIT |
| STACK-S8 | 42 | 427 | 12 | 12 | 6 | 4326 | 1965 | 658.0000 | 653.5600 | 0.006950 | 220.21s | TIME_LIMIT |
| STACK-S9 | 42 | 506 | 12 | 12 | 7 | 3078 | 1285 | 780.9999 | 772.0920 | 0.011631 | 260.11s | TIME_LIMIT |

## S8 与新 S9 对比

| 指标 | S8 | 新 S9 |
|---|---:|---:|
| seed | 42 | 42 |
| SKU 总需求量 | 427 | 506 |
| subtask 数 | 12 | 12 |
| task 数 | 12 | 12 |
| 实际命中 stack | 6 | 7 |
| 变量数 | 4326 | 3078 |
| route arc | 1965 | 1285 |
| Cmax | 658.0 | 780.9999 |
| gap | 0.00695 | 0.011631 |
| runtime | 220.21s | 260.11s |

## 运行命令

### STACK-S1

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S1 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 60 --mip-gap 0.01 --candidate-stack-topk 8 --candidate-station-topk-per-stack 2 --route-pickup-neighbor-limit 0 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s1_range_1_5_batch_1_3_station2_pruned_r0
```

### STACK-S2

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S2 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 60 --mip-gap 0.01 --candidate-stack-topk 8 --candidate-station-topk-per-stack 2 --route-pickup-neighbor-limit 5 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s2_range_1_5_batch_1_3_station2_pruned_r5
```

### STACK-S3

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S3 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 100 --mip-gap 0.01 --candidate-stack-topk 8 --candidate-station-topk-per-stack 1 --route-pickup-neighbor-limit 0 --disable-all-prune --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s3_range_1_5_batch_1_3_station1_no_path_prune
```

### STACK-S4

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S4 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 150 --mip-gap 0.01 --candidate-stack-topk 3 --candidate-station-topk-per-stack 2 --route-pickup-neighbor-limit 5 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s4_range_1_5_batch_1_3_station2_pruned_r5
```

### STACK-S5

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S5 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 150 --mip-gap 0.01 --candidate-stack-topk 3 --candidate-station-topk-per-stack 2 --route-pickup-neighbor-limit 5 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s5_range_1_5_batch_1_3_station2_pruned_r5
```

### STACK-S6

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S6 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 150 --mip-gap 0.01 --candidate-stack-topk 3 --candidate-station-topk-per-stack 2 --route-pickup-neighbor-limit 0 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s6_range_1_5_batch_1_3_station2_pruned_r0
```

### STACK-S7

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S7 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 220 --mip-gap 0.01 --candidate-stack-topk 3 --candidate-station-topk-per-stack 1 --route-pickup-neighbor-limit 5 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s7_baseline_gurobi_log_t220
```

### STACK-S8

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S8 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 220 --mip-gap 0.01 --candidate-stack-topk 3 --candidate-station-topk-per-stack 2 --route-pickup-neighbor-limit 5 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s8_range_1_5_batch_1_3_station2_pruned_r5
```

### STACK-S9

```bash
/usr/local/bin/python3 experiments/run_gurobi_benchmark18_suite.py --scales STACK-S9 --runtime-config-json experiments/configs/stacked_single_block_runtime_configs.json --seed 42 --time-limit 260 --mip-gap 0.01 --candidate-stack-topk 7 --candidate-station-topk-per-stack 2 --route-pickup-neighbor-limit 5 --disable-warm-start --disable-warm-start-sp4 --output-dir result/stacked_single_block_runtime_s9_seed42_qty_1_5_batch_3_5_stack211111_cand7_station2_r5_t260
```

## 调试结论

- S9 固定 `seed=42` 后，不能靠换 seed 拉高 Cmax。
- 原 S9 的 `total_order_qty=254`，明显低于 S8 的 `427`，所以 Cmax 低于 S8。
- S9 通过把 `bom_batch_quantity_range` 从 `[1,3]` 提高到 `[3,5]`，将总需求量提高到 `506`。
- S9 通过 `bom_colocated_stack_counts=[2,1,1,1,1,1]` 和 `candidate-stack-topk=7`，实际命中 stack 调整为 `7`。
- S9 最终变量数 `3078`，低于 S8 的 `4326`；但因总需求更高，Cmax 达到 `780.9999`。
- S7 和 S9 的 gap 略高于 `0.01`，按本轮口径作为 accepted-with-note 结果保留。
