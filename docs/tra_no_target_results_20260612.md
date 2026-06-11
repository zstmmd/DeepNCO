# TRA no-target experiment notes

Date: 2026-06-12

All runs below were executed with target guidance disabled:

- `known_target_guidance=false`
- `target_table_fastpath=false`
- `target_probe_case_presets=false`
- `global_target_probe=false`
- `fixgurobi_final_validation=false`
- `fixgurobi_enable_best_obj_stop=false`

## Full M1-M9 baseline

Output directory: `tmp/tra_m1_m9_no_target_cache_20260611`

Common parameters:

```powershell
--tra-revolving-mode
--revolving-layer-order AUTO
--revolving-mark-limit 20
--max-iters 4
--fixgurobi-time-limit-sec 700
--fixgurobi-coarse-time-limit-sec 60
--no-fixgurobi-accept-first-improvement
--no-fixgurobi-final-validation
--no-fixgurobi-enable-best-obj-stop
--no-known-target-guidance
--no-target-table-fastpath
--no-target-probe-case-presets
--no-global-target-probe
--no-resource-global-decomp-repair
--no-resource-candidate-pool-log
--compact-tra-summary-json
```

| Case | TRA Cmax | Gurobi target | Runtime (s) | Verification |
|---|---:|---:|---:|---|
| M1 | 489 | 489 | 2161.26 | PASS |
| M2 | 557 | 546 | 2421.20 | PASS |
| M3 | 558 | 558 | 1954.96 | PASS |
| M4 | 634 | 630 | 1849.39 | PASS |
| M5 | 681 | 679 | 3338.86 | PASS |
| M6 | 687 | 687 | 2168.24 | PASS |
| M7 | 726 | 708 | 2711.54 | PASS |
| M8 | 726 | 725 | 2634.72 | PASS |
| M9 | 736 | 731 | 3521.15 | PASS |

## Targeted M5/M6/M8 tuning

These runs keep the same no-target scoring rule and are intended for reporting faster medium-scale TRA settings.

| Case | Parameters | TRA Cmax | Gurobi Cmax | TRA time (s) | Gurobi time (s) | Speedup vs Gurobi | Verification |
|---|---|---:|---:|---:|---:|---:|---|
| M5 | `Y,Y / i2 / t400 / coarse25 / accept_first=true` | 681 | 679 | 1145.61 | 2098.58 | 45.41% | PASS |
| M6 | `AUTO / i4 / t500 / coarse40 / accept_first=false` | 687 | 687 | 1290.81 | 2288.03 | 43.58% | PASS |
| M8 | `Y / i1 / t900 / coarse40 / accept_first=false` | 726 | 725 | 658.41 | 2527.37 | 73.95% | PASS |

Notes:

- M6 matches the Gurobi Cmax while running 43.58% faster.
- M8 uses the accepted `Cmax=726` tolerance and is much faster than Gurobi.
- M5 cannot currently reach `Cmax=679` within the 30-50% faster target. The best fast setting found is `Cmax=681`, 45.41% faster. Historical `Cmax=679` runs require a long final `YZ/X` exact solve and are close to the Gurobi runtime.
