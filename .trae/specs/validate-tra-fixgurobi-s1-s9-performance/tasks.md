# Tasks
- [x] Task 1: 确认验收输入与 baseline 数据
  - [x] SubTask 1.1: 确认 S1-S9 使用的 runtime config、seed、Gurobi baseline Cmax 和 Gurobi baseline runtime 来源
  - [x] SubTask 1.2: 确认 TRA-FixGurobi 运行命令使用 LKH/OR-Tools 初始化，不使用 SP4 MIP
  - [x] SubTask 1.3: 确认 `Gurobi/global_xyzu.py` 未被修改

- [x] Task 2: 运行 S1-S9 TRA-FixGurobi 并收集结果
  - [x] SubTask 2.1: 批量运行 `STACK-S1` 到 `STACK-S9` 的 TRA-FixGurobi
  - [x] SubTask 2.2: 导出每个 case 的 best solution、native Cmax、global-style replay Cmax 和 runtime
  - [x] SubTask 2.3: 记录每个 case 的搜索日志、候选拒绝原因和 best 更新来源

- [x] Task 3: 注入 global Gurobi 校验完整解
  - [x] SubTask 3.1: 对 S1-S9 导出的完整解执行 fixed `XYZU` 注入 global Gurobi
  - [x] SubTask 3.2: 对比 TRA Cmax 与 global 注入 Cmax，要求差异在容差内
  - [x] SubTask 3.3: 对注入失败或 Cmax 不一致的 case 生成失败诊断，包括 infeasible 阶段、固定变量范围和关键约束信息

- [x] Task 4: 处理 `TRA Cmax < Gurobi baseline Cmax` 的 case
  - [x] SubTask 4.1: 筛选所有 TRA Cmax 小于 Gurobi baseline Cmax 的 case
  - [x] SubTask 4.2: 对这些 case 强制复核 fixed `XYZU` 注入结果
  - [x] SubTask 4.3: 若注入违反 global 约束，定位搜索过程中未排除该候选的路径并提出修复点
  - [x] SubTask 4.4: 若注入可行且 Cmax 一致，标记为 baseline 配置、剪枝参数或最优性证明需复核

- [x] Task 5: 验证搜索过程中非法候选不会进入 best
  - [x] SubTask 5.1: 检查 `FixGurobiEvaluator`、`ResourceValidator` 和 ALNS engine 的 accepted/best 更新路径
  - [x] SubTask 5.2: 确认不可注入或违反 coverage/route/station 约束的候选被 `F_raw=inf`、hard reject 或 validation rollback 排除
  - [x] SubTask 5.3: 若发现非法候选可能进入 `best_validated`，补充修复任务并重新验证

- [x] Task 6: runtime 快 20% 验收
  - [x] SubTask 6.1: 对齐 TRA-FixGurobi runtime 与 Gurobi baseline runtime 的计时口径
  - [x] SubTask 6.2: 计算每个 case 的 `TRA runtime / Gurobi runtime`
  - [x] SubTask 6.3: 标记是否满足 `TRA runtime <= 0.8 * Gurobi runtime`
  - [x] SubTask 6.4: 对不达标 case 输出主要耗时来源和下一步优化建议

- [x] Task 7: 生成最终验收报告
  - [x] SubTask 7.1: 输出 S1-S9 汇总表，包含 Cmax、注入状态、runtime、speedup 和验收结论
  - [x] SubTask 7.2: 输出所有异常 case 的诊断与处理建议
  - [x] SubTask 7.3: 明确说明是否满足 “Cmax 与 Gurobi 一致、非法更优解可注入核查、runtime 快 20%” 三项要求

- [x] Task 8: 修复非法候选进入 best 的风险
  - [x] SubTask 8.1: 将 `ResourceValidator` 中 BOM arrival window 与 kitting/order time-window 违反转为 hard reject
  - [x] SubTask 8.2: 修复 `FixGurobiEvaluator` 的 `WARM_START_FALLBACK` 分支，确保 span/deadline overrun 先被拒绝
  - [x] SubTask 8.3: 增加缺省 cfg 属性保护，避免 validator 测试因缺少配置属性提前失败

- [x] Task 9: 用当前代码重新跑 S1-S9 TRA-FixGurobi 与注入验收
  - [x] SubTask 9.1: 使用当前代码重新生成 S1-S9 TRA-FixGurobi 输出，不复用旧的 6/22 搜索输出
  - [x] SubTask 9.2: 对最新输出执行 fixed `XYZU` 注入 global Gurobi
  - [x] SubTask 9.3: 重新计算 Cmax 一致性、TRA<baseline 注入复核和 runtime 快 20% 结果

- [x] Task 10: 解释 S3/S4 为什么 Gurobi baseline 未找到更优解
  - [x] SubTask 10.1: 收集 `STACK-S3`、`STACK-S4` 的 Gurobi baseline 求解状态、best bound、MIP gap、time limit、剪枝参数和 runtime config
  - [x] SubTask 10.2: 将 TRA 的 221/256 解 fixed `XYZU` 注入同一 baseline config，确认是否在完全相同 global 参数下可行
  - [x] SubTask 10.3: 检查 baseline 是否存在 time limit 未证明最优、剪枝图排除了 TRA 路径、候选站点/堆垛 topk 不一致或 warm start/配置口径不一致
  - [x] SubTask 10.4: 输出 S3/S4 差异结论，明确 Gurobi 未找到更优解的原因和需要复核的 baseline 假设

- [x] Task 11: 对比 Gurobi 最优结构与 TRA best 结构
  - [x] SubTask 11.1: 导出或解析 Gurobi baseline/复核最优解的 X/Y/Z/U 结构
  - [x] SubTask 11.2: 对比 TRA best 与 Gurobi 解的订单拆分、站点分配、station rank、stack/tote 命中、SORT/FLIP 模式和机器人路径
  - [x] SubTask 11.3: 统计 S1-S9 中 TRA 慢或 Cmax 未达目标 case 的共同结构差异
  - [x] SubTask 11.4: 形成可转化为算子的结构模式清单，标注每个模式影响的层和预期收益

- [x] Task 12: 新增或增强 TRA-FixGurobi 搜索算子
  - [x] SubTask 12.1: 基于 Task11 的结构模式选择最小必要算子，优先增强收敛速度而不改变 global 约束语义
  - [x] SubTask 12.2: 实现算子或候选生成策略，并接入现有 ALNS/operator profile
  - [x] SubTask 12.3: 确保所有新候选仍经过 FixGurobi/global validation hard gate，不合法解不能进入 best
  - [x] SubTask 12.4: 为新增算子补充聚焦测试或可重复验证脚本

- [x] Task 13: 重新运行增强后的 S1-S9 最终验收（已执行，验收未通过）
  - [x] SubTask 13.1: 使用增强算子重新运行 S1-S9 TRA-FixGurobi
  - [x] SubTask 13.2: 对最终 best 执行 fixed `XYZU` global 注入，TRA Cmax 与 global Cmax 一致
  - [x] SubTask 13.3: 对比 Gurobi 认可目标 Cmax；结果未满足 TRA 最优解与 Gurobi 一致
  - [x] SubTask 13.4: 对比 runtime；结果未满足每个 case `TRA runtime <= 0.8 * Gurobi runtime`
  - [x] SubTask 13.5: 输出未达标 case 的剩余瓶颈和下一轮优化建议

- [x] Task 14: 更新最终验收报告（最终验收未通过）
  - [x] SubTask 14.1: 记录 S3/S4 Gurobi baseline 差异解释
  - [x] SubTask 14.2: 记录新增算子、触发次数、接受次数和带来的 Cmax/runtime 改善
  - [x] SubTask 14.3: 输出 S1-S9 最终 Cmax、global 注入、runtime 和是否快 20% 的汇总结论

- [ ] Task 15: 修复未通过验收项并重新验证
  - [x] SubTask 15.1: 统一 Task13 final validation 与 fixed injection 口径，消除 injection `OPTIMAL` 但 final validation `FIXGUROBI_FAILED` 的歧义
  - [x] SubTask 15.2: 方法一，调整 global Gurobi 跑 S3/S4 的候选/剪枝/站点 topk 等运行参数，使 baseline 也能求到并认可 TRA fixed `XYZU` 中的更优解；若不能，输出具体阻断参数或约束
  - [x] SubTask 15.3: 方法二，审计 S3/S4 fixed replay 与完整 U/route 约束等价性，找出 full global 不存在或不等价的 U 边，并在 TRA 搜索、SP4 LKH、U 算子和导出前 hard gate 中 ban 掉
  - [ ] SubTask 15.4: 针对 S2/S6/S7/S8/S9 的目标 Cmax 差距增强 X/Y/Z/U 结构种子与 Z/noise split 算子，但所有候选必须通过 full global 边集检查
  - [ ] SubTask 15.5: 针对 S1/S2/S4/S5/S6/S8 的 runtime 未达标问题降低 exact gate 成本、增强 cheap gate 与缓存复用
  - [ ] SubTask 15.6: 重新运行 S1-S9，要求 TRA 最优解与 Gurobi 认可目标 Cmax 一致，且每个 case runtime 快 20%

- [ ] Task 16: Task15.4-15.6 失败后的补救验证（代码已实现并执行，双硬指标仍未通过）
  - [x] SubTask 16.1: 修复 baseline fixed-route structure replay 在 S1 超预算无返回的问题，为 route edge audit 与 fixed global solve 分别加耗时记录、超时保护和失败诊断
  - [x] SubTask 16.2: 为 S2/S6/S7/S8/S9 增加可进入 TRA 搜索的 Gurobi 结构种子/受控释放种子，而不是直接用 baseline 导出替代 TRA best
  - [x] SubTask 16.3: 为 S1/S2/S4/S5/S6/S8 增加 cheap gate/cache/early stop：station load/order span 下界、重复 route signature 缓存、route edge audit 编译缓存
  - [x] SubTask 16.4: 重跑 S1-S9 并输出 TRA 搜索、fixed `XYZU` 注入、full global route edge audit、Cmax 与 runtime 0.8 阈值完整表（结果 FAIL：标准A 2/9、标准B 3/9、A+B 同时满足 0/9；9 个 case 最终验证全部被 route edge audit 硬门拒绝，HARD_GATE_REJECT，missing_edge_count 10-27，report: `result/task16_final_20260623/task16_acceptance_report.md`）

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 2
- Task 6 depends on Task 1 and Task 2
- Task 7 depends on Task 3, Task 4, Task 5, and Task 6
- Task 8 depends on Task 5
- Task 9 depends on Task 8
- Task 7 also depends on Task 9
- Task 10 depends on Task 9
- Task 11 depends on Task 10
- Task 12 depends on Task 11
- Task 13 depends on Task 12
- Task 14 depends on Task 13
- Task 15 depends on Task 14
- Task 16 depends on Task 15
