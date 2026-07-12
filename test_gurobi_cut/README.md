# M4 Order Time Window Cut Experiments

This folder is isolated from the production solver code. It tests the M4
`[22,16x5] + bq[3,3] + gap50` configuration and compares order time-window and
model simplification strategies.

## Baseline

Known 800s baseline:

- `cmax = 884`
- `bound = 874.227`
- `gap = 1.195%`
- `solve = 800.39s`
- `vars = 22,993`
- `constr = 63,846`
- `u_arc = 17,531`
- `total_qty = 765`
- `span/deadline overrun = 0`

The baseline result directory is:

```text
result/middle_bomseq_m4_seed42_t800_g01_sku22_16x5_bq33_gap50_noordertw_hit3_support20_copy1_stationtop1_slotlex_lbcuts_routearrlinear_r0_formal_20260702
```

## Files

- `base_config_gap50.json`: local copy of the M4 gap50 instance config.
- `run_m4_time_window_cut_experiments.py`: standalone experiment runner.
- `summarize_results.py`: result aggregator and report generator.
- `results/`: generated experiment outputs.
- `summary.csv`: generated compact metrics table.
- `time_window_simplification_report.md`: generated analysis report.

## Quick Run

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 test_gurobi_cut/run_m4_time_window_cut_experiments.py --time-limit 120 --force
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 test_gurobi_cut/summarize_results.py
```

## Full 800s Run

This can take several hours for all cases:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 test_gurobi_cut/run_m4_time_window_cut_experiments.py --time-limit 800 --force
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 test_gurobi_cut/summarize_results.py
```

## Strategy Coverage

- Strategy 1, fixed station choice: native solver config via `candidate_station_topk_per_stack`.
- Strategy 2, restricted stack candidates: native solver config via `candidate_stack_topk` and `max_candidate_stacks_per_order`.
- Strategy 3, route pattern: current production solver has no path-pattern variable formulation; this experiment uses `integrate_u_route=False` as a route-relaxed proxy only.
- Strategy 4, fixed slot order: uses existing non-CLI `GlobalXYZUConfig.fixed_work_units_by_order_slot`.
- Strategy 5, stack-level SKU/tote aggregation: reported as a static variable-count estimate, not solved, because it requires a new formulation.
