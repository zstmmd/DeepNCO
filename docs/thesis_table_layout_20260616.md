# Thesis Table Layout Aligned with Shao Jie and Transportation Science

This note converts the current experiment outputs into table layouts close to:

- Shao Jie thesis Chapter 7: parameter table, Gurobi vs integrated exact/decomposition, metaheuristic comparison, R3/G3 phased rules.
- Zhen et al. (Transportation Science, 2023): instance group table, method comparison with objective, runtime, gap, acceleration.

## 1. Instance Parameter Tables

### Table A-1 Medium-Scale Instance Parameters

Source: `result/medium_scale_instance_params_20260616.csv`

| Case | Map | Robots | Stations | Totes | Orders | SKUs | SKU Qty Range | Stack Count |
|---|---|---:|---:|---:|---:|---:|---|---:|
| GUROBI-M1 | 4x5 | 5 | 4 | 172 | 6 | 42 | 11-13 | 30 |
| GUROBI-M2 | 4x5 | 5 | 5 | 184 | 7 | 44 | 13-15 | 33 |
| GUROBI-M3 | 4x5 | 5 | 5 | 197 | 7 | 49 | 13-15 | 36 |
| GUROBI-M4 | 4x5 | 5 | 5 | 225 | 7 | 56 | 15-17 | 41 |
| GUROBI-M5 | 4x5 | 5 | 5 | 237 | 8 | 58 | 16-18 | 44 |
| GUROBI-M6 | 4x5 | 5 | 5 | 249 | 8 | 61 | 17-18 | 47 |
| GUROBI-M7 | 4x5 | 5 | 5 | 253 | 8 | 62 | 17-20 | 48 |
| GUROBI-M8 | 4x5 | 5 | 5 | 261 | 8 | 64 | 18-20 | 50 |
| GUROBI-M9 | 4x5 | 5 | 5 | 265 | 8 | 65 | 18-20 | 51 |

### Table A-2 Large-Scale Instance Parameters

Source: `result/large_scale_instance_params_20260616.csv`

| Case | Map | Robots | Stations | Totes | Orders | SKUs | Lines/Order | SKU Qty Range | Stack Count |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| L1 | 5x8 | 6 | 5 | 350 | 15 | 80 | 3-4 | 18-20 | 60 |
| L2 | 6x8 | 7 | 6 | 450 | 20 | 100 | 3-4 | 15-17 | 75 |
| L3 | 6x9 | 8 | 7 | 600 | 25 | 120 | 3-4 | 9-11 | 95 |
| L4 | 7x9 | 10 | 8 | 750 | 30 | 140 | 4-5 | 4-6 | 120 |
| L5 | 8x10 | 12 | 10 | 900 | 40 | 180 | 4-5 | 1-2 | 145 |
| L6 | 9x10 | 14 | 12 | 1100 | 50 | 220 | 4-5 | 1-2 | 175 |
| L7 | 10x11 | 16 | 14 | 1300 | 60 | 330 | 5-6 | 1-2 | 205 |
| L8 | 11x11 | 18 | 14 | 1500 | 80 | 430 | 5-6 | 1-2 | 235 |
| L9 | 12x12 | 20 | 15 | 1700 | 100 | 540 | 5-6 | 1-2 | 265 |

Use this as the equivalent of Shao Jie's small/medium/large parameter tables and TS Table 2.

## 2. Medium-Scale Tables

### Table M-1 Gurobi and TRA-FixGurobi Results

This follows Shao Jie's "Gurobi and MP-MD" table style. Use it for exact-repair evidence, not for every M case.

Source: `result/medium_scale_fixgurobi_evidence_20260616.csv`

| Case | Gurobi Cmax | Gurobi Time | TRA-Fix Cmax | TRA-Fix Time | Gap |
|---|---:|---:|---:|---:|---:|
| M1 | 489 | 1118.51 | 489 | 984.70 | 0.00% |
| M2 | 546 | 1667.09 | 546 | 990.00 | 0.00% |
| M3 | 558 | 1994.37 | 558 | 525.17 | 0.00% |
| M5 | 679 | 2098.58 | 679 | 777.77 | 0.00% |
| M6 | 687 | 2288.03 | 688 | 1591.57 | 0.15% |
| M8 | 725 | 2527.37 | 725 | 948.33 | 0.00% |

Recommended note: M4, M7, and M9 are omitted from this exact-repair table because the latest targeted FixGurobi evidence has not been rerun under the same profile.

### Table M-2 Gurobi and TRA-Fast Results

This follows the TS method-comparison style: objective, time, and relative gap.

Source: `result/medium_scale_current_summary_20260616.csv`

| Case | Gurobi Cmax | Gurobi Time | TRA-Fast Cmax | TRA-Fast Time | Gap |
|---|---:|---:|---:|---:|---:|
| M1 | 489 | 1115.63 | 496 | 46.05 | 1.43% |
| M2 | 546 | 1664.95 | 594 | 46.69 | 8.79% |
| M3 | 558 | 1992.81 | 598 | 47.47 | 7.17% |
| M4 | 630 | 2087.37 | 685 | 57.67 | 8.73% |
| M5 | 679 | 2097.25 | 715 | 62.59 | 5.30% |
| M6 | 687 | 2287.37 | 744 | 36.40 | 8.30% |
| M7 | 708 | 2481.76 | 775 | 90.86 | 9.46% |
| M8 | 725 | 2525.89 | 771 | 64.19 | 6.34% |
| M9 | 731 | 3452.09 | 754 | 72.29 | 3.15% |
| Avg. | 637 | 2191.00 | 693 | 58.69 | 6.50% |

Recommended note: runtime is not required to be monotone by case because TRA-Fast stops by quality/cap and different cases trigger different calibration paths.

## 3. Large-Scale Tables

### Table L-1 Gurobi Bound and TRA-Fast/Exact Results

This follows Shao Jie's large-scale Gurobi/MP-MD table, but should explicitly mark exact fallback.

Sources:

- `result/large_quantity_bounds_l1_l9_summary_20260615.csv`
- `result/large_quantity_repeat2000_l1_l9_summary_20260615.csv`

Recommended columns:

| Case | Relax LB | Relax Time | Mem-Bound Status | TRA-Fast Cmax | TRA-Fast Time | TRA-Exact Cmax | TRA-Exact Time | Exact Status |
|---|---:|---:|---|---:|---:|---:|---:|---|

Do not hide `TIMEOUT_seeded_fallback`; it is essential for a defensible thesis table.

### Table L-2 R3/G3 and TRA-Fast Comparison

This follows Shao Jie's R3/G3 comparison tables and TS Table 4's rule-vs-integrated structure.

Source for current accepted large results: `result/large_quantity_l1_l9_fast_portfolio_greedy_20260615/large_algorithm_suite_summary.csv`

Recommended columns:

| Case | R3 Cmax | R3 Time | G3 Cmax | G3 Time | TRA-Fast Cmax | TRA-Fast Time | Selected | Gap vs R3 | Gap vs G3 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|

This table is important because current TRA-Fast usually selects R3 or G3. It makes the algorithm contribution transparent.

## 4. Sensitivity Tables

To match Shao Jie's later tables and TS Section 5.3, add four sensitivity groups after the main comparison tables:

1. Change warehouse length/width.
2. Change order count.
3. Change tote/SKU count.
4. Change robot/station count.

Recommended columns:

| Scenario | Map | Orders | SKUs | Totes | Robots | Stations | Cmax | Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|

These should be run with TRA-Fast or the final proposed method only, and each row should average 3-5 seeds.

## 5. Current Missing Items Before Thesis Finalization

- Rebuild Table L-2 from current large quantity results if the source table is mixed with older non-quantity-strengthened runs.
- Rerun `layered_mip4` on the current L1-L9 quantity-strengthened instances, or exclude it from the final main large-scale table.
- Add 3-5 seeds for the main M/L tables if time allows. This will reduce concerns about nonmonotone runtime.
- Keep TRA-Fast and TRA-FixGurobi in separate tables. They answer different questions: fast feasible solution versus exact local repair.
