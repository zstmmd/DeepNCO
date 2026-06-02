# GUROBI-M Suite Calibration

This script searches runtime scale configs for `GUROBI-M1` through `GUROBI-M9`.

It enforces the acceptance chain against the previous accepted case:

- problem scale increases by at least one of orders, SKUs, totes, or active stacks, and none decrease
- `model_cmax` increases
- wall-clock runtime increases
- `model_gap <= 0.01`
- runtime is below the configured time limit
- `M1` additionally targets the configured runtime window, default `950s..1200s`

The Gurobi route-pruning policy is fixed to:

- `route_arc_prune=True`
- `enable_route_load_interval_arc_prune=True`
- `enable_route_time_window_arc_prune=False`
- `enable_route_directional_arc_prune=False`
- `route_pickup_neighbor_limit=0`
- warm start disabled
- scale-adaptive candidate prune disabled

Dry-run candidate preview:

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' experiments/calibrate_gurobi_m_suite.py `
  --stages 9 `
  --max-candidates-per-stage 16 `
  --dry-run `
  --output-dir result/gurobi_m_calibration_dryrun
```

Full calibration:

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' experiments/calibrate_gurobi_m_suite.py `
  --stages 9 `
  --time-limit-sec 3600 `
  --mip-gap 0.01 `
  --m1-min-runtime-sec 950 `
  --m1-max-runtime-sec 1200 `
  --max-candidates-per-stage 16 `
  --output-dir result/gurobi_m_calibration_full
```

Outputs:

- `candidate_results.jsonl`: every solved candidate, append-only for resume
- `selected_chain.json`: accepted M-chain
- `selected_chain.csv`: accepted M-chain table
- `selected_problem_configs.json`: configs to copy into `CreateOFSProblem` after validation
