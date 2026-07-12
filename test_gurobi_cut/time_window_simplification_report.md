# M4 Order Time Window Simplification Experiment

## Fixed 800s Baseline

- cmax=884.0, bound=874.227, gap=1.195%, solve=800.39s, vars=22993, constr=63846, u_arc=17531, total_qty=765, span/deadline overrun=0/0.

## Short-Horizon Experiment Results

| Case | cmax | bound | gap | solve(s) | vars | constr | u_arc | span/deadline | Δsolve vs short baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_no_tw | 967 | 874.219 | 9.722% | 120.12 | 22993 | 63846 | 17531 | 0/0 | 0 |
| s1_station_fixed_top1_tw | 895 | 874.220 | 2.422% | 120.12 | 23017 | 63927 | 17531 | 0/0 | -0.00 |
| s1_station_relaxed_top2_tw | 1229 | 874.095 | 63.216% | 120.03 | 44082 | 126942 | 37966 | 229/44 | -0.09 |
| s2_stack_top2_tw | 895 | 874.220 | 2.433% | 120.11 | 23017 | 63927 | 17531 | 0/0 | -0.01 |
| s2_stack_top4_tw | 895.000 | 874.220 | 2.433% | 120.13 | 23017 | 63927 | 17531 | 0/0 | 0.01 |
| s3_route_relaxed_proxy_tw | 1217 | 766.769 | 12.988% | 120.02 | 4044 | 7481 | 0 | 0/0 | -0.10 |
| s4_fixed_slot_order_tw | 909 |  |  | 0.21 | 23017 | 64274 | 0 | 0/0 | -119.91 |
| tw_on_no_cut | 895 | 874.219 | 2.434% | 120.12 | 23017 | 63927 | 17531 | 0/0 | -0.00 |

## 800s Time-Window Confirmation

- Time windows enabled at 800s: cmax=884, bound=874.222, gap=1.196%, solve=800.27s, vars=23017, constr=63927, u_arc=17531, span/deadline=0/0.
- Compared with the fixed no-time-window 800s baseline, time windows add 24 vars and 81 constraints; gap changes by 0.002 percentage points.

## Findings

### Order Time Windows

At the 120s screening horizon, enabling order time windows increases the model only slightly (24 vars, 81 constraints), but improves the incumbent and gap substantially.

- No time windows: cmax=967, gap=9.722%, bound=874.219.
- Time windows enabled: cmax=895, gap=2.434%, bound=874.219.

Interpretation: for this gap50 M4 instance, order time windows act as useful tightening rather than harmful overhead.

### Strategy 1: Fixed Station Choice

`candidate_station_topk_per_stack=1` matches the time-window case: vars=23017, u_arc=17531, gap=2.422%.
Relaxing to station top2 blows up the model to vars=44082, constraints=126942, u_arc=37966, and gap=63.216%; it also creates time-window violations.

Conclusion: station top1 is already essential and should remain fixed for M4.

### Strategy 2: Restrict Stack Candidates

Top2 and top4 have the same model size as the time-window baseline (top2 vars=23017, top4 vars=23017).
The effective candidate set is already about three stacks/order due to the colocated inventory profile, so additional top-k limits do not materially change this instance.

### Strategy 3: Route Pattern Proxy

Disabling integrated route decisions drops vars to 4044 and constraints to 7481, but the bound falls to 766.769 and gap remains 12.988%.
Conclusion: a real path-pattern formulation may reduce size, but simply removing route arcs is too weak to compare with the full model.

### Strategy 4: Fixed Slot Order

The naive BOM/SKU chunk fixed-slot case returned `WARM_START_FALLBACK` with model status infeasible/no incumbent. It is not a valid convergence improvement.
Conclusion: fixed slot order needs to be generated from a feasible warm solution or allow local swaps; naive SKU chunks over-constrain the model.

## Strategy 5 Static Estimate

Stack-level SKU/tote aggregation was not solved because it requires a new model formulation. The estimate below uses the current variable mix and replaces tote-level SKU selection with stack-SKU variables.

- Conservative stack-SKU formulation: reduces 228 vars (0.99%).
- If tote interval `sort` is also approximated at stack service level: reduces 1653 vars (7.18%).
- Optimistic cover-only lower bound: reduces 2637 vars (11.46%).

## Overall Recommendation

Keep order time windows enabled for this M4 family unless the 800s confirmation contradicts the 120s screening result.
Keep station top1 fixed. Do not relax to top2/topall.
Further stack-topk tightening is not useful on the current colocated hit3/support20 profile.
Route pattern and stack aggregation require new formulations; current proxy/estimate suggests route formulation is the more important target than SKU/tote aggregation.
