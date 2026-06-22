# BOM-Split M-Suite Experiment Notes (2026-06-16)

## Purpose

This experiment builds BOM-split variants on top of the existing `GUROBI-M1` to `GUROBI-M9` cases:

| Profile | Target subtasks/orders |
|---|---:|
| Split-1 | 1.2 |
| Split-2 | 1.5 |
| Split-3 | 2.0 |

Because the original M cases have only 6-8 orders, Split-1 cannot always hit 1.2 exactly. The script records both `target_split_ratio` and `actual_split_ratio`.

## Script

Main script:

```text
experiments/run_bom_split_m_suite.py
```

The script does not modify `problemDto/createInstance.py`. It patches `CreateOFSProblem.generate_problem_by_scale()` inside the experiment process, so `TRAOptimizer`, `TRA-FixGurobi`, and direct Gurobi all receive the same split case.

Current generation mode is `config_split_v2`: split cases are derived as runtime M-case configs first, then the standard instance generator rebuilds orders and colocated inventory for the split orders. This replaces the earlier post-hoc split prototype, which split orders after the original M inventory had already been generated and could make TRA's SP3 heuristic path inconsistent.

## Outputs

Each run writes:

```text
bom_split_case_stats.csv
bom_split_algorithm_results.csv
bom_split_comparison_with_current_m.csv
bom_split_run_config.json
```

`bom_split_comparison_with_current_m.csv` starts with the current M-series baseline table, then appends Split-1/2/3 rows.

## Smoke Evidence

Dry-run checked M1 and M9 split construction:

```text
result/bom_split_m1_m9_dryrun_20260616/bom_split_case_stats.csv
```

Full v2 dry-run checked all 27 split cases:

```text
result/bom_split_m_suite_dryrun_full_v2_20260616/bom_split_case_stats.csv
```

Short algorithm smoke checked `GUROBI-M1-SPLIT-1`:

```text
result/bom_split_m1_split1_smoke_20260616
```

This used intentionally short limits and is not a formal numerical result:

| Algorithm | Status | Cmax | Runtime (s) |
|---|---|---:|---:|
| Gurobi | TIME_LIMIT | 793 | 26.66 |
| TRA-Fast | ok | 555 | 18.18 |
| TRA-FixGurobi | ok | 714 | 110.81 |

An additional TRA-Fast-only smoke after summary-column adjustment:

```text
result/bom_split_m1_split1_tra_fast_smoke2_20260616
```

## Formal Run Commands

Full sequential run for all 27 split cases:

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' experiments/run_bom_split_m_suite.py `
  --cases GUROBI-M1 GUROBI-M2 GUROBI-M3 GUROBI-M4 GUROBI-M5 GUROBI-M6 GUROBI-M7 GUROBI-M8 GUROBI-M9 `
  --splits SPLIT-1 SPLIT-2 SPLIT-3 `
  --algorithms gurobi tra_fixgurobi tra_fast `
  --gurobi-time-limit-sec 3600 `
  --gurobi-mip-gap 0.01 `
  --resume `
  --output-root result/bom_split_m_suite_full_v2_20260616
```

Recommended batched run, one split profile at a time:

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' experiments/run_bom_split_m_suite.py `
  --splits SPLIT-1 `
  --algorithms gurobi tra_fixgurobi tra_fast `
  --gurobi-time-limit-sec 3600 `
  --gurobi-mip-gap 0.01 `
  --resume `
  --output-root result/bom_split_m_suite_split1_v2_20260616
```

Resume/rebuild only the comparison table after partial runs:

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' experiments/run_bom_split_m_suite.py `
  --rebuild-comparison-only `
  --output-root result/bom_split_m_suite_full_v2_20260616
```

Run a single formal case, useful for long Gurobi runs:

```powershell
& 'D:/anaconda/envs/deepnco_ml_312/python.exe' experiments/run_bom_split_m_suite.py `
  --cases GUROBI-M1 `
  --splits SPLIT-1 `
  --algorithms gurobi tra_fixgurobi tra_fast `
  --gurobi-time-limit-sec 3600 `
  --gurobi-mip-gap 0.01 `
  --resume `
  --output-root result/bom_split_m_suite_full_v2_20260616
```

The script defaults to `--resume`, so interrupted formal runs can be restarted with the same command.

## Formal V2 Results

Source:

```text
result/bom_split_m_suite_full_v2_20260616/bom_split_comparison_with_current_m.csv
```

Completed rows:

| Case | Split | Gurobi cmax | TRA-FixGurobi cmax | TRA-Fast cmax | Best TRA cmax | Best TRA algorithm | Best TRA vs Gurobi gap | Gurobi gap | Gurobi s | TRA-FixGurobi s | TRA-Fast s |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| GUROBI-M1-SPLIT-1 | Split-1 | 458 | 458 | 554 | 458 | tra_fixgurobi | 0.00% | 0.019347 | 3606.96 | 1122.97 | 23.49 |
| GUROBI-M1-SPLIT-2 | Split-2 | 421 | 420 | 582 | 420 | tra_fixgurobi | -0.24% | 0.131397 | 3607.11 | 3479.61 | 23.83 |

For Split-2, Gurobi reached the 3600-second limit with a relatively large gap. The comparison is therefore against the limited-time Gurobi incumbent, not a proven optimum.

Audit for the TRA-FixGurobi row passes coverage and makespan consistency:

```text
result/bom_split_m_suite_full_v2_20260616/GUROBI-M1-SPLIT-1/tra_fixgurobi/GUROBI-M1-SPLIT-1/best_solution_export/best_solution_audit.json
```

## Current M Baseline

| Case | Gurobi cmax | TRA cmax | TRA vs Gurobi gap | Gurobi gap | Gurobi s | TRA s |
|---|---:|---:|---:|---:|---:|---:|
| GUROBI-M1 | 489 | 489 | 0.00% | 0.000221 | 1115.63 | 825.94 |
| GUROBI-M2 | 546 | 546 | 0.00% | 0.000307 | 1664.95 | 990.00 |
| GUROBI-M3 | 558 | 558 | 0.00% | 0.009555 | 1992.81 | 450.48 |
| GUROBI-M4 | 630 | 630 | 0.00% | 0.009641 | 2087.37 | 1603.56 |
| GUROBI-M5 | 679 | 679 | 0.00% | 0.003238 | 2097.25 | 777.77 |
| GUROBI-M6 | 687 | 687 | 0.00% | 0.000048 | 2287.37 | 1591.57 |
| GUROBI-M7 | 708 | 708 | 0.00% | 0.003626 | 2481.76 | 1348.67 |
| GUROBI-M8 | 725 | 726 | 0.14% | 0.003022 | 2525.89 | 948.43 |
| GUROBI-M9 | 731 | 731 | 0.00% | 0.005605 | 3452.09 | 1216.08 |
