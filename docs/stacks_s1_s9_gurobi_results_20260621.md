# STACKS-S1-S9 Gurobi Results - 2026-06-21

来源：`experiments/configs/stacks_s1_s9_gurobi_baseline_20260620.json` 与采用结果目录下的 `run_details.json`。

| case | BOM数 | 每BOM SKU类型数 | 总SKU数 | robot | station | tote | stack | batch qty | 单件每SKU用量 | status | Cmax | gap | runtime(s) | 变量数 | route arc | 采用结果目录 |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| STACKS-S1 | 2 | 7,7 | 18 | 2 | 2 | 48 | 8 | U(1,3) | U(1,5) | OPTIMAL | 103 | 0.000229 | 15.37 | 1495 | 796 | `result/stacks_s1_s9_user_repro_20260620/s1` |
| STACKS-S2 | 2 | 10,10 | 32 | 2 | 2 | 48 | 8 | U(1,3) | U(1,5) | OPTIMAL | 158 | 0.000000 | 55.26 | 1885 | 1036 | `result/stacks_s1_s9_user_repro_20260620/s2` |
| STACKS-S3 | 2 | 14,14 | 46 | 2 | 2 | 52 | 8 | U(1,3) | U(1,5) | OPTIMAL | 239 | 0.000184 | 73.40 | 2441 | 976 | `result/stacks_s1_s9_user_repro_20260620/s3_pruned_focus2` |
| STACKS-S4 | 2 | 18,18 | 60 | 2 | 2 | 56 | 8 | U(1,3) | U(1,5) | OPTIMAL | 258 | 0.000000 | 88.19 | 3587 | 1984 | `result/stacks_s1_s9_user_repro_20260620/s4` |
| STACKS-S5 | 4 | 7,7,7,7 | 74 | 3 | 2 | 60 | 8 | U(1,3) | U(1,5) | OPTIMAL | 269 | 0.009613 | 129.64 | 2794 | 1317 | `result/stacks_s1_s9_user_repro_20260620/s5` |
| STACKS-S6 | 4 | 10,10,10,10 | 88 | 3 | 2 | 60 | 8 | U(1,3) | U(1,5) | OPTIMAL | 386 | 0.006762 | 130.68 | 2196 | 1097 | `result/stacks_s1_s9_user_repro_20260620/s6_stack1111_no_prune_r0` |
| STACKS-S7 | 4 | 15,15,15,15 | 102 | 3 | 2 | 62 | 8 | U(1,3) | U(1,5) | OPTIMAL | 524 | 0.007803 | 159.06 | 3489 | 1443 | `result/stacks_s1_s9_user_repro_20260620/s7_focus2_t300` |
| STACKS-S8 | 4 | 18,18,18,18 | 116 | 3 | 2 | 64 | 8 | U(1,3) | U(1,5) | OPTIMAL | 660 | 0.009990 | 101.67 | 3555 | 1443 | `result/stacks_s1_s9_user_repro_20260620/s8_stack2111_copy1` |
| STACKS-S9 | 6 | 7,7,7,7,7,7 | 130 | 3 | 2 | 64 | 8 | U(3,5) | U(1,5) | OPTIMAL | 779 | 0.009107 | 65.92 | 3078 | 1285 | `result/stacks_s1_s9_user_repro_20260620/s9_focus3_h005_cand7_r5_t300` |

## 备注

- S6 的采用结果为 `s6_stack1111_no_prune_r0`，runtime 只比 S5 多约 1.03s。
- S8 的采用结果为 `s8_stack2111_copy1`。
- S9 的采用结果为 `s9_focus3_h005_cand7_r5_t300`；同目录组里另有 `s9_focus2_cand7_r5_t300` 得到 Cmax=778，但不采用，因为当前基准表固定为 Cmax=779。
