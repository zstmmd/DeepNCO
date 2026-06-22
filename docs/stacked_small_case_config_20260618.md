# Stacked Small Case Configuration - 2026-06-18

本文件记录当前单储区堆垛式小规模算例配置、求解结果和运行命令。所有命令均使用：

```powershell
D:/anaconda/envs/deepnco_ml_312/python.exe
```

## 地图口径

当前 `map_size: [1, 1]` 不再表述为 block，而表述为**单储区/单堆垛区地图**。程序内部对应离散路径网格约为 `9 x 11`，用于机器人 Manhattan 距离计算；真实储位规模由 `stack_count` 和 `warehouse_block_height` 表示。

| 参数 | 当前口径 |
|---|---|
| 储区形态 | 单储区堆垛式 |
| 内部地图参数 | `map_size = [1, 1]` |
| 内部离散网格 | `9 x 11` |
| stack 最大高度 | 8 |
| 工作台位置 | 地图边界侧 |
| 路径模型 | 网格 Manhattan 距离 |

## 核准算例与 SKU10 对照

| 指标 | STACK-S1 | STACK-S1M | STACK-S2 | STACK-BOM2-SKU10 |
|---|---:|---:|---:|---:|
| 用途 | 基础小规模 | S1/S2 中间档 | 较难小规模 | 每 BOM SKU=10 敏感性测试 |
| BOM 数 | 2 | 2 | 4 | 2 |
| 每 BOM SKU 类型数 | 7 | 8 | 8 | 10 |
| 总 SKU 数 | 18 | 20 | 40 | 24 |
| robot 数 | 2 | 2 | 3 | 2 |
| station 数 | 2 | 2 | 2 | 2 |
| 配置 tote 数 | 44 | 44 | 48 | 44 |
| 实际生成 tote 数 | 44 | 44 | 48 | 44 |
| stack 数 | 8 | 8 | 8 | 8 |
| 平均 stack 高度 | 5.5 | 5.5 | 6.0 | 5.5 |
| 最大 stack 高度 | 8 | 8 | 6 | 8 |
| batch quantity | U(1,5) | U(1,5) | U(1,5) | U(1,5) |
| 单件每 SKU 用量 | U(5,10) | U(5,10) | U(5,10) | U(5,10) |
| 库存布局 | 默认库存 + mid_low_redundancy | 默认库存 + mid_low_redundancy | BOM 局部集中存放 | 默认库存 + mid_low_redundancy |
| BOM 局部 stack 组 | - | - | [2,2,2,2] | - |
| 需求 SKU 平均覆盖 stack 数 | 4.0714 | 3.1875 | 1.75 | 1.6 |
| 需求 SKU 最大覆盖 stack 数 | 7 | 5 | 2 | 2 |
| 命中 stack 数 | 3 | 6 | 5 | 4 |
| 命中 stack IDs | [38,42,57] | [38,50,57,65,68,87] | [38,42,50,57,83] | [38,42,83,87] |
| 任务数 | 6 | 8 | 10 | 5 |
| 变量数 | 2967 | 2705 | 1788 | 581 |
| route arc 数 | 1360 | 1182 | 765 | 156 |
| passX 数 | 120 | 112 | 138 | 40 |
| Cmax | 513 | 565 | 1433 | 953 |
| runtime | 46.9729s | 131.2722s | 28.7192s | 0.4271s |
| gap | 0.009841 | 0.009219 | 0.008108 | 0.000054 |
| status | OPTIMAL | OPTIMAL | OPTIMAL | OPTIMAL |
| warmstart | disabled | disabled | disabled | disabled |

结论：原始 `STACK-BOM2-SKU10` 在默认库存下命中过 8 个 stack，求解时间达到 231.7s；按“命中 stack 数控制为 4”的要求改为 BOM 局部集中存放后，命中 stack 数降为 4，求解时间降到 0.43s。说明每 BOM SKU 类型数本身会增加需求量和 SKU 决策，但真正放大 Gurobi 难度的是需求 SKU 分布到多少个 stack 以及由此产生的路径组合。

生成器修复：`bom_colocated_inventory` 现在补 filler tote 时，若非 BOM stack 已满或不存在，会继续补 BOM stack，直到达到配置 tote 数；默认库存和 BOM 局部库存都会至少生成 `ceil(stack_count * 5.5)` 个 tote，避免后续算例平均 stack 高度低于 5.5。

## 当前配置片段

```json
{
  "STACK-S1": {
    "map_size": [1, 1],
    "warehouse_block_height": 8,
    "resources": [2, 2, 44],
    "data": [2, 18],
    "bom_complexity": [7, 1],
    "target_stack_count": 8,
    "exact_disjoint_bom_sku_count": 7,
    "exact_disjoint_bom_sku_quantity_range": [5, 10],
    "bom_batch_quantity_unit": 1,
    "bom_batch_quantity_range": [1, 5],
    "exact_demand_sku_strategy": "",
    "bom_colocated_inventory": true,
    "bom_colocated_stack_counts": [2, 2],
    "bom_colocated_disjoint_stack_groups": true,
    "bom_colocated_support_multiplier": 1.2,
    "bom_colocated_sku_copy_count": 2,
    "bom_colocated_chunked_by_stack": true
  },
  "STACK-S1M": {
    "map_size": [1, 1],
    "warehouse_block_height": 8,
    "resources": [2, 2, 44],
    "data": [2, 20],
    "bom_complexity": [8, 1],
    "target_stack_count": 8,
    "exact_disjoint_bom_sku_count": 8,
    "exact_disjoint_bom_sku_quantity_range": [5, 10],
    "bom_batch_quantity_unit": 1,
    "bom_batch_quantity_range": [1, 5],
    "exact_demand_sku_strategy": "mid_low_redundancy",
    "bom_colocated_inventory": false
  },
  "STACK-S2": {
    "map_size": [1, 1],
    "warehouse_block_height": 8,
    "resources": [3, 2, 48],
    "data": [4, 40],
    "bom_complexity": [8, 1],
    "target_stack_count": 8,
    "exact_disjoint_bom_sku_count": 8,
    "exact_disjoint_bom_sku_quantity_range": [5, 10],
    "bom_batch_quantity_unit": 1,
    "bom_batch_quantity_range": [1, 5],
    "exact_demand_sku_strategy": "",
    "bom_colocated_inventory": true,
    "bom_colocated_stack_counts": [2, 2, 2, 2],
    "bom_colocated_disjoint_stack_groups": true,
    "bom_colocated_support_multiplier": 1.2,
    "bom_colocated_sku_copy_count": 2,
    "bom_colocated_chunked_by_stack": true
  },
  "STACK-BOM2-SKU10": {
    "map_size": [1, 1],
    "warehouse_block_height": 8,
    "resources": [2, 2, 44],
    "data": [2, 24],
    "bom_complexity": [10, 1],
    "target_stack_count": 8,
    "exact_disjoint_bom_sku_count": 10,
    "exact_disjoint_bom_sku_quantity_range": [5, 10],
    "bom_batch_quantity_unit": 1,
    "bom_batch_quantity_range": [1, 5],
    "exact_demand_sku_strategy": "mid_low_redundancy",
    "bom_colocated_inventory": false
  }
}
```

## 运行命令

### STACK-S1

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' --scales STACK-S1 --seed 42 --time-limit 150 --mip-gap 0.01 --candidate-stack-topk 3 --route-pickup-neighbor-limit 5 --candidate-station-topk-per-stack 1 --disable-warm-start --disable-order-time-windows --big-m-time 10000 --route-big-m-time 10000 --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' --output-dir 'result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s1_station1_bigm10000'
```

### STACK-S1M

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' --scales STACK-S1M --seed 42 --time-limit 150 --mip-gap 0.01 --candidate-stack-topk 3 --max-candidate-stacks-per-order 6 --route-pickup-neighbor-limit 5 --candidate-station-topk-per-stack 1 --disable-warm-start --disable-order-time-windows --big-m-time 10000 --route-big-m-time 10000 --gurobi-mip-focus 2 --gurobi-heuristics 0.3 --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' --output-dir 'result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s1m_2bom8sku_stack6'
```

### STACK-S2

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' --scales STACK-S2 --seed 42 --time-limit 150 --mip-gap 0.01 --candidate-stack-topk 3 --route-pickup-neighbor-limit 5 --candidate-station-topk-per-stack 2 --disable-warm-start --disable-order-time-windows --big-m-time 10000 --route-big-m-time 10000 --gurobi-mip-focus 2 --gurobi-heuristics 0.3 --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' --output-dir 'result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s2_after_fill_fix_coldsku'
```

### STACK-BOM2-SKU10

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' 'experiments/run_gurobi_benchmark18_suite.py' --scales STACK-BOM2-SKU10 --seed 42 --time-limit 300 --mip-gap 0.01 --candidate-stack-topk 3 --max-candidate-stacks-per-order 6 --route-pickup-neighbor-limit 5 --candidate-station-topk-per-stack 1 --disable-warm-start --disable-order-time-windows --big-m-time 10000 --route-big-m-time 10000 --gurobi-mip-focus 2 --gurobi-heuristics 0.3 --runtime-config-json 'experiments/configs/stacked_single_block_runtime_configs.json' --output-dir 'result/stacked_batch_u1_5_20260618/bom2_sku10_hit4_after_fill_fix'
```

## 输出文件

| case | run details | solution dump |
|---|---|---|
| STACK-S1 | `result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s1_station1_bigm10000/run_details.json` | `result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s1_station1_bigm10000/STACK-S1/gurobi_solution_export/best_solution_full_dump.txt` |
| STACK-S1M | `result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s1m_2bom8sku_stack6/run_details.json` | `result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s1m_2bom8sku_stack6/STACK-S1M/gurobi_solution_export/best_solution_full_dump.txt` |
| STACK-S2 | `result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s2_after_fill_fix_coldsku/run_details.json` | `result/stacked_batch_u1_5_20260618/gurobi_no_warm_fixedM_s2_after_fill_fix_coldsku/STACK-S2/gurobi_solution_export/best_solution_full_dump.txt` |
| STACK-BOM2-SKU10 | `result/stacked_batch_u1_5_20260618/bom2_sku10_hit4_after_fill_fix/run_details.json` | `result/stacked_batch_u1_5_20260618/bom2_sku10_hit4_after_fill_fix/STACK-BOM2-SKU10/gurobi_solution_export/best_solution_full_dump.txt` |
