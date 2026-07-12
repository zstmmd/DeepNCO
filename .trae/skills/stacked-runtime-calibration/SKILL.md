---
name: "stacked-runtime-calibration"
description: "Calibrates STACK-S1..S9 Gurobi runtime cases. Invoke when tuning stacked warehouse benchmark configs, pruning, hit stacks, Cmax, gap, or result reports."
---

# Stacked Runtime Calibration

## Scope

Use this skill when tuning dense stacked warehouse benchmark cases in this repository, especially `STACK-S1` to `STACK-S9` under:

```text
experiments/configs/stacked_single_block_runtime_configs.json
experiments/run_gurobi_benchmark18_suite.py
result/stacked_single_block_runtime*
```

The goal is to produce a calibrated series of Gurobi benchmark cases with controlled runtime, monotonic Cmax, reproducible seed, and documented final parameters/results.

## Current Accepted Series

The current accepted report is:

```text
experiments/configs/stacked_single_block_runtime_report.md
```

The accepted config is:

```text
experiments/configs/stacked_single_block_runtime_configs.json
```

Final retained result directories:

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

## Calibration Rules

Follow these rules unless the user explicitly changes the target:

- Keep `seed=42` across the accepted series.
- Keep dense stacked layout: each case should have average tote/stack greater than `5.5`.
- Keep `target_stack_count=8`.
- Keep `exact_order_sku_quantity_range=[1,5]` by default.
- Keep `bom_batch_quantity_range=[1,3]` by default, except final `STACK-S9`, which uses `[3,5]` to make total demand exceed S8.
- Disable warm start for accepted benchmark runs:

```bash
--disable-warm-start --disable-warm-start-sp4
```

- Prefer pruning and inventory layout changes before changing map size or robot/station counts.
- Do not change seed just to improve Cmax or gap unless the user explicitly allows it.

## Workflow

1. Inspect current config and retained results.

```bash
jq '.configs["STACK-S9"], .results["STACK-S9"]' experiments/configs/stacked_single_block_runtime_configs.json
find result -maxdepth 1 -type d -name 'stacked_single_block_runtime*' | sort
```

2. Use dry-run before expensive solves.

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
  --output-dir result/stacked_single_block_runtime_s9_probe_dryrun
```

3. Run full Gurobi only after dry-run demand and scale look correct.

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

4. Extract result metrics from `summary.csv`, `run_details.json`, and `best_solution_full_dump.txt`.

```bash
jq '.[0] | {model_cmax, model_best_bound, model_gap, runtime_sec, model_var_count_total, u_arc_count}' \
  result/stacked_single_block_runtime_s9_seed42_qty_1_5_batch_3_5_stack211111_cand7_station2_r5_t260/run_details.json

awk '/^subtask_id=.*order_id=/{nsub++}
     /^task_id=.*stack_id=/{task++; if (match($0,/stack_id=[0-9]+/)) {s=substr($0,RSTART+9,RLENGTH-9); stacks[s]=1}}
     END{printf "subtask=%d task=%d hit_stack=%d\n", nsub, task, length(stacks)}' \
  result/stacked_single_block_runtime_s9_seed42_qty_1_5_batch_3_5_stack211111_cand7_station2_r5_t260/STACK-S9/gurobi_solution_export/best_solution_full_dump.txt
```

5. Update `experiments/configs/stacked_single_block_runtime_report.md` after any accepted change.

## Final S8 And S9 Reference

| Metric | S8 | New S9 |
|---|---:|---:|
| seed | 42 | 42 |
| SKU total demand | 427 | 506 |
| subtask count | 12 | 12 |
| task count | 12 | 12 |
| actual hit stacks | 6 | 7 |
| variable count | 4326 | 3078 |
| route arc | 1965 | 1285 |
| Cmax | 658.0 | 780.9999 |
| gap | 0.00695 | 0.011631 |
| runtime | 220.21s | 260.11s |

## Tuning Notes

- If Cmax is too low, first inspect `total_order_qty`, `min_order_sku_qty`, and `max_order_sku_qty`.
- If S9 is below S8, increasing `bom_batch_quantity_range` is more direct than changing seed.
- If hit stack count is too high, reduce `bom_colocated_stack_counts` and then constrain `candidate-stack-topk`.
- If variable count is too high, inspect `route_arc`, `sort`, `carry`, `noise`, `passX`, `route_load`, `route_owner`, and `route_time`.
- S7 and S9 may be accepted with a small gap above 1% when runtime and Cmax sequence are otherwise satisfactory.

## Cleanup Policy

After accepting a run, delete intermediate `result/stacked_single_block_runtime*` directories and keep only the final adopted result directories listed in this skill and report.
