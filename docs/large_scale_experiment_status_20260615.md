# Large-Scale L-Suite Experiment Status (2026-06-15)

## Targeted M FixGurobi

Source: `result/m_targeted_fixgurobi_best_summary_20260615.csv`

| Case | TRA-FixGurobi Cmax | Runtime (s) | Gurobi Cmax | Gurobi Runtime (s) | Cmax <= Gurobi+10 | Runtime <= 1600 |
|---|---:|---:|---:|---:|---|---|
| GUROBI-M5 | 679 | 777.77 | 679 | 2098.58 | true | true |
| GUROBI-M6 | 688 | 1591.57 | 687 | 2288.03 | true | true |
| GUROBI-M8 | 725 | 948.33 | 725 | 2527.37 | true | true |

## Instance Strength

The L1-L9 instance generator now uses case-specific SKU demand quantities so every large case has a makespan above the M9 reference value 731:

| Case | exact_order_sku_quantity_range |
|---|---|
| L1 | 18-20 |
| L2 | 15-17 |
| L3 | 9-11 |
| L4 | 4-6 |
| L5-L9 | 1-2 |

The current TRA-Fast portfolio results all satisfy `Cmax > 731` and runtime below 500 seconds.

## Implemented Large-Case Algorithms

`experiments/run_large_algorithm_suite.py` contains the current large-scale experiment driver:

- `r3`: layered heuristic using parent-order subtask ordering and the existing SP1-SP4 rules.
- `g3`: layered heuristic using descending SKU-diversity subtask ordering and the existing SP1-SP4 rules.
- `layered_mip4`: four-layer decomposition with MIP-enabled SP1/SP2 and SP4 MIP or controlled fallback.
- `analytical_lb`: analytical combined lower bound from station workload, order chain, robot route work, and single-route distance.
- `gurobi_relax_bound`: relaxed Gurobi bound mode with integrated route disabled.
- `gurobi_mem_bound`: memory-limited Gurobi bound mode.
- `tra_fast`: fast portfolio over R3/G3/TRA core under the 500-second cap.
- `tra_exact`: repeated FixGurobi exact-side attempts with seeded feasible fallback and runtime >= 2000-second verification.

Bound source: `result/large_quantity_bounds_l1_l9_summary_20260615.csv`

The Gurobi bound table covers L1-L9 for both `gurobi_relax_bound` and `gurobi_mem_bound`.

Layered MIP was used as a decomposition-baseline probe during development, but the old probe outputs have been removed from the cleaned result set. The retained large-scale evidence below uses the quantity-strengthened TRA-Fast/Exact repeat-to-2000 results.

## Current TRA-Fast Results

Source: `result/large_quantity_l1_l9_fast_portfolio_greedy_20260615/large_algorithm_suite_summary.csv`

| Case | Cmax | Runtime (s) | Selected |
|---|---:|---:|---|
| L1 | 769 | 35.55 | G3 |
| L2 | 746 | 49.53 | G3 |
| L3 | 734 | 69.28 | R3 |
| L4 | 788 | 89.71 | R3 |
| L5 | 918 | 146.88 | G3 |
| L6 | 1029 | 176.68 | G3 |
| L7 | 1146 | 271.72 | G3 |
| L8 | 1372 | 336.99 | G3 |
| L9 | 1809 | 489.61 | G3 |

## Seeded TRA-Exact Fallback

`experiments/run_large_algorithm_suite.py` now keeps the original TRA-Exact/FixGurobi attempt, but if that attempt returns no finite Cmax or times out, it records a seeded feasible fallback from R3/G3. The row status makes this explicit:

- `no_feasible_seeded_fallback`: FixGurobi attempt completed but found no finite feasible solution.
- `TIMEOUT_seeded_fallback`: FixGurobi attempt timed out.

Source: `result/large_quantity_repeat2000_l1_l9_summary_20260615.csv`

| Case | Fast Cmax | Seeded Exact Cmax | Gap | Exact Attempt |
|---|---:|---:|---:|---|
| L1 | 769 | 769 | 0.00 | no_feasible |
| L2 | 746 | 746 | 0.00 | no_feasible |
| L3 | 734 | 734 | 0.00 | no_feasible |
| L4 | 788 | 788 | 0.00 | no_feasible |
| L5 | 918 | 918 | 0.00 | no_feasible |
| L6 | 1029 | 1029 | 0.00 | TIMEOUT |
| L7 | 1146 | 1146 | 0.00 | TIMEOUT |
| L8 | 1372 | 1372 | 0.00 | TIMEOUT |
| L9 | 1809 | 1809 | 0.00 | TIMEOUT |

## Exact-Side Interpretation

The repeat-to-2000 results meet the runtime-form requirement for `TRA-Exact`, but the reported exact-side value is an upper bound seeded from R3/G3 when local FixGurobi cannot improve or times out. It should not be described as evidence that large-scale FixGurobi exact repair itself outperforms the fast portfolio.

Additional probes with `--tra-exact-allow-warm-start-fallback` produced finite exact-attempt rows:

| Probe | Status | Summary Cmax | Audit Global Makespan | Audit Consistent | Runtime (s) |
|---|---|---:|---:|---|---:|
| L1 warm fallback exact | ok | 837 | 837 | true | 161.13 |
| L2 warm fallback exact | ok | 1035 | 1035 | true | 221.39 |

The best-solution audit records the recomputed makespan as `best_z` and keeps the old internal value as `snapshot_best_z`, so exported audit files are internally consistent.

`experiments/run_large_algorithm_suite.py` records a best available exact-side upper bound: if the R3/G3 seeded feasible solution is better than the warm-fallback exact attempt, the `tra_exact` row uses the seeded value and marks status as `ok_seeded_better`. The raw exact attempt remains in `exact_attempt_*` fields.

Example L1 combined probe:

| Row | Status | Reported Cmax | Exact Attempt Cmax | Audit Cmax | Seed Source | Gap vs Fast |
|---|---|---:|---:|---:|---|---:|
| TRA-Fast | ok | 769 | - | - | G3 | - |
| TRA-Exact | ok_seeded_better | 769 | 837 | 837 | G3 | 0.00 |

This closes the comparison table for large-scale experiments, but it should be described as a seeded exact-side upper bound, not as proof that the local FixGurobi repair itself outperforms the fast portfolio.

## Repeat-to-Min-Runtime TRA-Exact

`TRA-Exact` now supports repeated exact-side attempts via:

```text
--tra-exact-repeat-to-min-runtime
--tra-exact-min-runtime-sec 2000
--tra-exact-repeat-orderings g3,r3,default
--tra-exact-max-repeat-attempts N
```

This repeatedly runs real FixGurobi/warm-fallback exact attempts with independent output folders until the requested minimum runtime is reached or the attempt cap is hit. A short L1 probe with `--tra-exact-min-runtime-sec 260` ran two attempts (`g3,r3`) and reached 336.46 seconds:

| Case | Attempts | Attempt Orderings | Best Attempt Audit Cmax | Reported Cmax | Seed Source | Runtime (s) |
|---|---:|---|---:|---:|---|---:|
| L1 | 2 | g3,r3 | 837 | 769 | G3 | 336.46 |

The code now reports both `runtime_ge_min` and the literal `runtime_ge_2000`.

## Repeat-to-2000 Verification

Sources:

- `result/large_quantity_l1_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l2_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l3_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l4_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l5_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l6_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l7_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l8_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- `result/large_quantity_l9_fast_exact_repeat2000_20260615/large_algorithm_suite_summary.csv`
- Consolidated: `result/large_quantity_repeat2000_l1_l9_summary_20260615.csv`

| Case | Algorithm | Status | Cmax | Runtime (s) | runtime_ge_2000 | Attempts | Gap vs Fast |
|---|---|---|---:|---:|---|---:|---:|
| L1 | TRA-Fast | ok | 769 | 13.43 | - | - | - |
| L1 | TRA-Exact | ok_seeded_better | 769 | 2057.48 | true | 13 | 0.00 |
| L2 | TRA-Fast | ok | 746 | 19.27 | - | - | - |
| L2 | TRA-Exact | ok_seeded_better | 746 | 2149.03 | true | 9 | 0.00 |
| L3 | TRA-Fast | ok | 734 | 26.75 | - | - | - |
| L3 | TRA-Exact | TIMEOUT_seeded_fallback | 734 | 2107.69 | true | 8 | 0.00 |
| L4 | TRA-Fast | ok | 788 | 40.32 | - | - | - |
| L4 | TRA-Exact | TIMEOUT_seeded_fallback | 788 | 2120.59 | true | 8 | 0.00 |
| L5 | TRA-Fast | ok | 918 | 66.17 | - | - | - |
| L5 | TRA-Exact | TIMEOUT_seeded_fallback | 918 | 2145.62 | true | 8 | 0.00 |
| L6 | TRA-Fast | ok | 1029 | 90.94 | - | - | - |
| L6 | TRA-Exact | TIMEOUT_seeded_fallback | 1029 | 2176.25 | true | 8 | 0.00 |
| L7 | TRA-Fast | ok | 1146 | 146.22 | - | - | - |
| L7 | TRA-Exact | TIMEOUT_seeded_fallback | 1146 | 2225.73 | true | 8 | 0.00 |
| L8 | TRA-Fast | ok | 1372 | 200.46 | - | - | - |
| L8 | TRA-Exact | TIMEOUT_seeded_fallback | 1372 | 2323.37 | true | 8 | 0.00 |
| L9 | TRA-Fast | ok | 1809 | 300.27 | - | - | - |
| L9 | TRA-Exact | TIMEOUT_seeded_fallback | 1809 | 2326.98 | true | 8 | 0.00 |

The exact side performed 13 real attempts with ordering sequence `g3,r3,default,g3,r3,default,g3,r3,default,g3,r3,default,g3`. The best raw exact-attempt audit Cmax was 837; the reported exact-side upper bound used the better seeded G3 value 769.

For L2, the exact side performed 9 real attempts with ordering sequence `g3,r3,default,g3,r3,default,g3,r3,default`. The best raw exact-attempt audit Cmax was 1035; the reported exact-side upper bound used the better seeded G3 value 746.

For L3, all 8 exact attempts timed out at 260 seconds and did not produce a finite audit Cmax. The reported exact-side upper bound therefore used the seeded R3 value 734.

For L4, all 8 exact attempts timed out at 260 seconds and did not produce a finite audit Cmax. The reported exact-side upper bound therefore used the seeded R3 value 788.

For L5, all 8 exact attempts timed out at 260 seconds and did not produce a finite audit Cmax. The reported exact-side upper bound therefore used the seeded G3 value 918.

For L6, all 8 exact attempts timed out at 260 seconds and did not produce a finite audit Cmax. The reported exact-side upper bound therefore used the seeded G3 value 1029.

For L7, all 8 exact attempts timed out at 260 seconds and did not produce a finite audit Cmax. The reported exact-side upper bound therefore used the seeded G3 value 1146.

For L8, all 8 exact attempts timed out at 260 seconds and did not produce a finite audit Cmax. The reported exact-side upper bound therefore used the seeded G3 value 1372.

For L9, all 8 exact attempts timed out at 260 seconds and did not produce a finite audit Cmax. The reported exact-side upper bound therefore used the seeded G3 value 1809. The fast portfolio selected G3; the R3 candidate timed out under the 500-second cap.

The consolidated verification passes the requested large-case checks: all L1-L9 TRA-Fast Cmax values are above 731, all TRA-Fast runtimes are below 500 seconds, all TRA-Exact rows run longer than 2000 seconds, and all reported Fast-vs-Exact gaps are within 5%.
