# STACKS-S1-S9 Sequential Configuration - 2026-06-19

本文档记录堆垛式小规模算例 `STACKS-S1` 到 `STACKS-S9` 的顺序校准结果。当前硬约束：

- Python 环境：`D:/anaconda/envs/deepnco_ml_312/python.exe`
- robot 容量固定使用系统默认值 `OFSConfig.ROBOT_CAPACITY = 8`，不允许通过算例配置修改。
- batch quantity：`U(1,3)`。
- BOM 数目标序列：`2,2,2,2,4,4,4,4,6`。
- 每 BOM SKU 类型数循环：`7,10,14,18,7,10,14,18,7`。
- 成功算例要求：`gap <= 0.01`，runtime 在 `20s-250s` 内递增，Cmax 递增。

## 当前进度

当前已顺序跑通并固定 `STACKS-S1` 到 `STACKS-S3`：

```text
runtime: 20.065897 < 27.195074 < 162.765204
Cmax:    94        < 119       < 273
```

`STACKS-S4` 到 `STACKS-S9` 尚未写入最终成功链，后续需要继续顺序校准。

## 成功算例参数

| case | BOM 数 | 每 BOM SKU 类型数 | 总 SKU 数 | robot | station | tote | stack | map_size | batch quantity | 单件每 SKU 用量 | 库存模式 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| STACKS-S1 | 2 | 7 | 18 | 2 | 2 | 44 | 8 | 1x1 | U(1,3) | U(1,3) | BOM 局部共址 |
| STACKS-S2 | 2 | 10 | 28 | 2 | 2 | 48 | 8 | 1x1 | U(1,3) | U(1,3) | BOM 局部共址 |
| STACKS-S3 | 2 | 14 | 36 | 2 | 2 | 48 | 8 | 1x1 | U(1,3) | U(3,5) | BOM 局部共址 |

## 求解结果

| case | status | Cmax | LB | gap | runtime | demanded SKU | total order qty | min order qty | max order qty | avg stack cover | 变量数 | route arc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STACKS-S1 | OPTIMAL | 94.000000 | 94.098000 | 0.000000 | 20.065897s | 14 | 43 | 11 | 32 | 1.000000 | 1305 | 532 |
| STACKS-S2 | OPTIMAL | 119.000000 | 118.086000 | 0.008447 | 27.195074s | 20 | 59 | 17 | 42 | 1.000000 | 1565 | 724 |
| STACKS-S3 | OPTIMAL | 273.000000 | 270.530000 | 0.009548 | 162.765204s | 28 | 169 | 53 | 116 | 1.000000 | 2475 | 1132 |

## 当前配置片段

完整配置以 `experiments/configs/stacked_single_block_runtime_configs.json` 为准。当前成功段核心字段如下：

```json
{
  "STACKS-S1": {
    "map_size": [1, 1],
    "warehouse_block_height": 8,
    "resources": [2, 2, 44],
    "data": [2, 18],
    "bom_complexity": [7, 1],
    "target_stack_count": 8,
    "exact_disjoint_bom_sku_count": 7,
    "exact_disjoint_bom_sku_quantity_range": [1, 3],
    "bom_batch_quantity_unit": 1,
    "bom_batch_quantity_range": [1, 3],
    "exact_demand_sku_strategy": "",
    "bom_colocated_inventory": true,
    "bom_colocated_stack_counts": [2, 2],
    "bom_colocated_sku_copy_count": 1,
    "bom_colocated_disjoint_stack_groups": true
  },
  "STACKS-S2": {
    "map_size": [1, 1],
    "warehouse_block_height": 8,
    "resources": [2, 2, 48],
    "data": [2, 28],
    "bom_complexity": [10, 1],
    "target_stack_count": 8,
    "exact_disjoint_bom_sku_count": 10,
    "exact_disjoint_bom_sku_quantity_range": [1, 3],
    "bom_batch_quantity_unit": 1,
    "bom_batch_quantity_range": [1, 3],
    "exact_demand_sku_strategy": "",
    "bom_colocated_inventory": true,
    "bom_colocated_stack_counts": [2, 2],
    "bom_colocated_sku_copy_count": 1,
    "bom_colocated_disjoint_stack_groups": true
  },
  "STACKS-S3": {
    "map_size": [1, 1],
    "warehouse_block_height": 8,
    "resources": [2, 2, 48],
    "data": [2, 36],
    "bom_complexity": [14, 1],
    "target_stack_count": 8,
    "exact_disjoint_bom_sku_count": 14,
    "exact_disjoint_bom_sku_quantity_range": [3, 5],
    "bom_batch_quantity_unit": 1,
    "bom_batch_quantity_range": [1, 3],
    "exact_demand_sku_strategy": "",
    "bom_colocated_inventory": true,
    "bom_colocated_stack_counts": [2, 2],
    "bom_colocated_sku_copy_count": 1,
    "bom_colocated_disjoint_stack_groups": true
  }
}
```

## 运行命令

### STACKS-S1

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' --scales STACKS-S1 --seed 42 --time-limit 300 --mip-gap 0.01 --candidate-stack-topk 3 --max-candidate-stacks-per-order 4 --route-pickup-neighbor-limit 5 --candidate-station-topk-per-stack 2 --disable-warm-start --disable-order-time-windows --big-m-time 10000 --route-big-m-time 10000 --gurobi-mip-focus 2 --gurobi-heuristics 0.3 --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' --output-dir 'result/stacks_s1_s9_sequential_20260619/s1'
```

### STACKS-S2

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' --scales STACKS-S2 --seed 42 --time-limit 300 --mip-gap 0.01 --candidate-stack-topk 3 --max-candidate-stacks-per-order 4 --route-pickup-neighbor-limit 5 --candidate-station-topk-per-stack 2 --disable-warm-start --disable-order-time-windows --disable-all-prune --big-m-time 10000 --route-big-m-time 10000 --gurobi-mip-focus 2 --gurobi-heuristics 0.3 --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' --output-dir 'result/stacks_s1_s9_sequential_20260619/s2'
```

### STACKS-S3

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' --scales STACKS-S3 --seed 42 --time-limit 300 --mip-gap 0.01 --candidate-stack-topk 3 --max-candidate-stacks-per-order 8 --route-pickup-neighbor-limit 8 --candidate-station-topk-per-stack 2 --disable-warm-start --disable-order-time-windows --big-m-time 1000 --route-big-m-time 1000 --gurobi-mip-focus 2 --gurobi-heuristics 0.3 --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' --output-dir 'result/stacks_s1_s9_sequential_20260619/s3_probe_pruned'
```

## 输出文件

| case | run details | summary | solution export |
|---|---|---|---|
| STACKS-S1 | `result/stacks_s1_s9_sequential_20260619/s1/run_details.json` | `result/stacks_s1_s9_sequential_20260619/s1/summary.csv` | `result/stacks_s1_s9_sequential_20260619/s1/STACKS-S1/gurobi_solution_export` |
| STACKS-S2 | `result/stacks_s1_s9_sequential_20260619/s2/run_details.json` | `result/stacks_s1_s9_sequential_20260619/s2/summary.csv` | `result/stacks_s1_s9_sequential_20260619/s2/STACKS-S2/gurobi_solution_export` |
| STACKS-S3 | `result/stacks_s1_s9_sequential_20260619/s3_probe_pruned/run_details.json` | `result/stacks_s1_s9_sequential_20260619/s3_probe_pruned/summary.csv` | `result/stacks_s1_s9_sequential_20260619/s3_probe_pruned/STACKS-S3/gurobi_solution_export` |

## 弃用尝试

| case/run | 结果 | 原因 |
|---|---|---|
| STACKS-S3 `[2,4]` normal | TIME_LIMIT, gap 约 0.0216 | 低用量导致 Cmax 太小，相对 gap 证明慢 |
| STACKS-S3 `[3,5]` + disable-all-prune + MIPFocus=3 | 运行过久，手动停止 | 人为加难过度，不适合作为 S3 |