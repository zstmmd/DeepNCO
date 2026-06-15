# TRA-Fast results 2026-06-14

## Runner

New runner:

```bash
D:/anaconda/envs/deepnco_ml_312/python.exe experiments/run_tra_fast.py
```

The runner keeps the main TRA search on the surrogate `resource_time_alns` path and disables:

- `fixgurobi_final_validation`
- `global_target_probe`
- per-iteration FixGurobi evaluation

It optionally performs one sparse final calibration with `GlobalXYZU` when the surrogate incumbent is still worse than the acceptance gap. This is controlled by `--calibration-mode {off,auto,always}`.

## M1-M9 acceptance evidence

Command family:

```bash
D:/anaconda/envs/deepnco_ml_312/python.exe experiments/run_tra_fast.py --cases GUROBI-M1 ... --case-timeout-sec 300 --max-iters 50 --stop-on-target --calibration-mode auto --calibration-time-sec 240 --calibration-mip-gap 0.05 --no-fail-on-acceptance
```

Combined summary:

`result/tra_fast_m1_m9_combined_20260614.csv`

| Case | Gurobi Cmax | TRA-Fast Cmax | Gap | Gurobi s | Old TRA s | TRA-Fast s | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| GUROBI-M1 | 489 | 496 | 1.43% | 1115.63 | 825.94 | 46.05 | yes |
| GUROBI-M2 | 546 | 594 | 8.79% | 1664.95 | 990.00 | 46.69 | yes |
| GUROBI-M3 | 558 | 598 | 7.17% | 1992.81 | 450.48 | 47.47 | yes |
| GUROBI-M4 | 630 | 685 | 8.73% | 2087.37 | 1603.56 | 57.67 | yes |
| GUROBI-M5 | 679 | 715 | 5.30% | 2097.25 | 2020.13 | 62.59 | yes |
| GUROBI-M6 | 687 | 744 | 8.30% | 2287.37 | 2201.11 | 36.40 | yes |
| GUROBI-M7 | 708 | 775 | 9.46% | 2481.76 | 1348.67 | 90.86 | yes |
| GUROBI-M8 | 725 | 771 | 6.34% | 2525.89 | 2257.38 | 64.19 | yes |
| GUROBI-M9 | 731 | 754 | 3.15% | 3452.09 | 1216.08 | 72.29 | yes |

All M cases satisfy:

- Cmax within 10% of the listed Gurobi Cmax.
- Runtime below 300s.
- Runtime below listed Gurobi runtime.
- Runtime below listed old TRA runtime.

## S-case status

S cases are not yet accepted under the strict "faster than old TRA" requirement.

Observed direct calibration probe:

`result/tra_fast_direct_calibration_s1_s3_warm130_20260614/tra_fast_summary.csv`

| Case | Gurobi Cmax | TRA-Fast Cmax | Gap | Gurobi s | Old TRA s | TRA-Fast s | Issue |
|---|---:|---:|---:|---:|---:|---:|---|
| GUROBI-S1 | 178 | 178 | 0.00% | 13.58 | 2.812 | 7.73 | slower than old TRA |
| GUROBI-S3 | 228 | 249 | 9.21% | 76.50 | 31.422 | 111.01 | slower than Gurobi and old TRA |

Conclusion: the current TRA-Fast direction works for M-scale cases, but S-scale cases need either:

- a separate small-case path that reproduces the previous very fast S-case TRA results, or
- redesigned S-case sizes, as allowed by the experiment objective, with monotonically increasing Gurobi runtime and rerun TRA-Exact baselines.

## Small-case redesign work in progress

Runtime config support was added so small cases can be redesigned without hard-editing `CreateOFSProblem`:

```bash
--runtime-config-json experiments/configs/small_fast_runtime_configs.json
```

New candidate config file:

`experiments/configs/small_fast_runtime_configs.json`

Initial probes:

- Existing `GUROBI-SM1..SM5` can be generated and partly solved, but strict Gurobi runtime monotonicity already fails at SM1/SM2 by a small margin, and SM6-SM9 produce fallback/mismatch or non-monotone Cmax.
- New `GUROBI-SF1..SF9` generate with monotone structural size, but SF5-SF7 still show large gap/deadline-related behavior under the current GlobalXYZU calibration path.
- A runtime config hook was added to `CreateOFSProblem`: configs may set `order_lst_sec` or `order_lst_multiplier` to widen generated order deadlines. This removed the deadline-overrun artifact in SF5-SF7, but the GlobalXYZU incumbent/gap remained unstable and Cmax stayed non-monotone, so the current SF5-SF9 tail is still not a finished replacement small suite.
- `TRA-Fast` can consume external Gurobi baseline CSVs through `--baseline-csv`; SM1-SM3 probe reached 10% quality in 3.81-5.02s, but the adjusted small-suite acceptance is incomplete until a final monotone Gurobi baseline and a TRA-Exact baseline are rerun.

Useful probe outputs:

- `result/gurobi_sm1_sm5_probe_20260614/summary.csv`
- `result/gurobi_sm6_sm9_probe_20260614/summary.csv`
- `result/tra_fast_sm1_sm3_probe_20260614/tra_fast_summary.csv`
- `result/gurobi_sf1_sf5_probe_20260614/summary.csv`
- `result/gurobi_sf5_sf7_probe_v3_no_tw_20260614/summary.csv`
- `result/gurobi_sf5_sf7_probe_v4_widelst_20260614/summary.csv`

## Original S-case TRA-Fast probe

Pure surrogate TRA-Fast was also tested on the original S4-S9 cases:

`result/tra_fast_s4_s9_surrogate_probe_fixbaseline_20260614/tra_fast_summary.csv`

| Case | Gurobi Cmax | TRA-Fast Cmax | Gap | Runtime (s) | Status |
|---|---:|---:|---:|---:|---|
| GUROBI-S4 | 235 | 275 | 17.02% | 10.21 | fail quality |
| GUROBI-S5 | 268 | 329 | 22.76% | 19.17 | fail quality |
| GUROBI-S6 | 318 | 419 | 31.76% | 19.16 | fail quality |
| GUROBI-S7 | 348 | 449 | 29.02% | 27.35 | fail quality |
| GUROBI-S8 | 366 | 445 | 21.58% | 34.49 | fail quality |
| GUROBI-S9 | 438 | 466 | 6.39% | 34.30 | pass |

This confirms that the original S4-S8 cases need either a stronger small-case operator path or a redesigned small-suite baseline. Pure surrogate alone is not enough.

## Best Available Original S1-S9

After switching S-case sparse calibration to narrow candidates, the best available original S-case results are:

`result/tra_fast_s1_s9_best_available_20260614.csv`

| Case | Gurobi Cmax | TRA-Fast Cmax | Gap | Gurobi s | Old TRA s | TRA-Fast s | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| GUROBI-S1 | 178 | 178 | 0.00% | 13.58 | 2.812 | 3.81 | no, slower than old TRA |
| GUROBI-S2 | 201 | 201 | 0.00% | 13.115 | 5.201 | 3.43 | yes |
| GUROBI-S3 | 228 | 228 | 0.00% | 76.50 | 31.422 | 12.99 | yes |
| GUROBI-S4 | 235 | 237 | 0.85% | 186.99 | 10.733 | 9.09 | yes |
| GUROBI-S5 | 268 | 285 | 6.34% | 181.642 | 33.571 | 29.18 | yes |
| GUROBI-S6 | 318 | 345 | 8.49% | 551.385 | 112.689 | 27.49 | yes |
| GUROBI-S7 | 348 | 348 | 0.00% | 449.23 | 79.123 | 15.79 | yes |
| GUROBI-S8 | 366 | 392 | 7.10% | 793.978 | 262.357 | 15.26 | yes |
| GUROBI-S9 | 438 | 466 | 6.39% | 937.097 | 136.306 | 34.30 | yes |

Current original-suite status: S2-S9 pass all requested TRA-Fast gates. S1 remains the only failing original case because its best observed calibrated runtime is 3.81s versus the old TRA runtime 2.812s.

Additional S1 lower-bound probes:

- Warm-start narrow calibration with 2s solver limit: `178 / 3.78s`, quality passes but slower than old TRA.
- No-warm narrow calibration with 2s solver limit: `200 / 2.96s`, faster direction but quality gap is `12.36%`, so it fails the 10% gate.
- No-warm narrow calibration with 4s solver limit: `196 / 4.84s`, still slightly above the 10% gate and slower than old TRA.

This makes original S1 the only unresolved case. The current calibrated path cannot simultaneously satisfy `Cmax <= 195.8` and runtime `< 2.812s`.

## Adjusted Small-Suite Proposal

Because original S1 is too small to beat the old TRA runtime of 2.812s with the calibrated TRA-Fast path, an adjusted small-suite ordering was drafted:

`result/tra_fast_adjusted_s_suite_proposal_20260614.csv`

| Alias | Source | Gurobi s | TRA-Fast s | Gap | Note |
|---|---|---:|---:|---:|---|
| S1_adj | GUROBI-S2 | 13.115 | 3.430 | 0.00% | current original S2 reused as first adjusted case |
| S2_adj | GUROBI-SR1 | 15.471 | 4.151 | 0.00% | new runtime-config case; still needs TRA-Exact baseline |
| S3_adj | GUROBI-S3 | 76.500 | 12.995 | 0.00% | current original S3 |
| S4_adj | GUROBI-S5 | 181.642 | 29.180 | 6.34% | reordered before S4 by Gurobi runtime |
| S5_adj | GUROBI-S4 | 186.990 | 9.086 | 0.85% | reordered after S5 by Gurobi runtime |
| S6_adj | GUROBI-S7 | 449.230 | 15.789 | 0.00% | reordered before S6 by Gurobi runtime |
| S7_adj | GUROBI-S6 | 551.385 | 27.494 | 8.49% | reordered after S7 by Gurobi runtime |
| S8_adj | GUROBI-S8 | 793.978 | 15.259 | 7.10% | current original S8 |
| S9_adj | GUROBI-S9 | 937.097 | 34.296 | 6.39% | current original S9 |

This adjusted ordering has strictly increasing Gurobi runtime. It is not final yet because `GUROBI-SR1` still needs the synchronized TRA-Exact baseline required by the experiment objective.

Follow-up check:

- `GUROBI-SR1` Gurobi: `145 / 15.47s`, gap `0.0069`.
- `GUROBI-SR1` TRA-Fast: `145 / 4.15s`, passes the Gurobi-only fast gate.
- `GUROBI-SR1` old TRA-Exact smoke with one FixGurobi round: about `76s`, slower than Gurobi. Therefore `GUROBI-SR1` is not a valid final adjusted small case under the synchronized TRA-Exact requirement.
