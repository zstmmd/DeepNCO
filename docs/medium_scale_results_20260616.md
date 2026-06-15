# Medium-Scale M-Suite Results (2026-06-16)

## Clean Result Tables

- Summary: `result/medium_scale_current_summary_20260616.csv`
- FixGurobi evidence: `result/medium_scale_fixgurobi_evidence_20260616.csv`

## TRA-Fast M1-M9

Source: `result/tra_fast_m1_m9_combined_20260614.csv`

| Case | Gurobi Cmax | Gurobi Runtime (s) | TRA-Fast Cmax | TRA-Fast Runtime (s) | Gap | Accept |
|---|---:|---:|---:|---:|---:|---|
| GUROBI-M1 | 489 | 1115.63 | 496 | 46.05 | 1.43% | true |
| GUROBI-M2 | 546 | 1664.95 | 594 | 46.69 | 8.79% | true |
| GUROBI-M3 | 558 | 1992.81 | 598 | 47.47 | 7.17% | true |
| GUROBI-M4 | 630 | 2087.37 | 685 | 57.67 | 8.73% | true |
| GUROBI-M5 | 679 | 2097.25 | 715 | 62.59 | 5.30% | true |
| GUROBI-M6 | 687 | 2287.37 | 744 | 36.40 | 8.30% | true |
| GUROBI-M7 | 708 | 2481.76 | 775 | 90.86 | 9.46% | true |
| GUROBI-M8 | 725 | 2525.89 | 771 | 64.19 | 6.34% | true |
| GUROBI-M9 | 731 | 3452.09 | 754 | 72.29 | 3.15% | true |

Interpretation: this is the speed-oriented M-suite result. All cases are within 10% of the Gurobi incumbent and much faster than Gurobi.

## TRA-FixGurobi Evidence

Sources:

- M1-M3: `result/m_tra_regression_20260614_rerun/m_tra_regression_summary.csv`
- M5: `result/m_targeted_fixgurobi_m5_nocompile_20260615/m_targeted_fixgurobi_summary.csv`
- M6: `result/m_targeted_fixgurobi_m6_m8_nocompile_20260615/m_targeted_fixgurobi_summary.csv`
- M8: `result/m8_nocompile_widecand_20260615/tra_gurobi_s1_s9_summary.csv`

| Case | TRA-FixGurobi Cmax | Runtime (s) | Gurobi Cmax | Gurobi Runtime (s) | Note |
|---|---:|---:|---:|---:|---|
| GUROBI-M1 | 489 | 984.70 | 489 | 1118.51 | reaches Gurobi Cmax |
| GUROBI-M2 | 546 | 990.00 | 546 | 1667.09 | reaches Gurobi Cmax |
| GUROBI-M3 | 558 | 525.17 | 558 | 1994.37 | reaches Gurobi Cmax |
| GUROBI-M5 | 679 | 777.77 | 679 | 2098.58 | reaches Gurobi Cmax |
| GUROBI-M6 | 688 | 1591.57 | 687 | 2288.03 | within Gurobi+10 |
| GUROBI-M8 | 725 | 948.33 | 725 | 2527.37 | reaches Gurobi Cmax |

Interpretation: this is the exact-repair evidence. M1-M3, M5, and M8 match the Gurobi Cmax; M6 is one unit worse but within the accepted `Gurobi+10` tolerance and under 1600 seconds.

## Current Gaps

- M4, M7, and M9 currently have clean TRA-Fast rows, but no latest targeted TRA-FixGurobi evidence table under the same tuned profile.
- TRA-Fast is strong as a speed baseline, but it should be presented separately from TRA-FixGurobi exact repair.
- For thesis tables, use `medium_scale_current_summary_20260616.csv` as the main M-suite table and cite the FixGurobi evidence table only for the exact-repair subset.
