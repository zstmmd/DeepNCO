# TRA-FixGurobi S1-S9 验证与性能验收 Spec

## Why
需要在不改变 `global_xyzu` baseline 约束语义的前提下，证明 TRA-FixGurobi 在 S1-S9 上的解满足 Gurobi 全局约束，并且运行时间相对 Gurobi baseline 至少快 20%。如果 TRA 搜索得到比 Gurobi baseline 更小的 Cmax，必须通过注入 global Gurobi 判断该解是否合法，并分析搜索过程中如何排除违反约束的解。

## What Changes
- 建立 S1-S9 TRA-FixGurobi 批量验收流程，输出每个 case 的 TRA Cmax、global 注入 Cmax、Gurobi baseline Cmax、runtime 和 speedup。
- 对任何 `TRA Cmax < Gurobi baseline Cmax` 的 case，自动执行 fixed `XYZU` 注入 global Gurobi 校验。
- 若注入 global 后不可行或 Cmax 口径不一致，记录违反约束原因、失败阶段、相关固定变量范围和搜索排除建议。
- 若注入 global 后可行且 Cmax 更小，标记为需要复核 baseline 最优性或 baseline 参数一致性，而不是直接判定 TRA 合法优于最优。
- 验证 TRA 搜索过程中被接受为 best 的候选必须经过 global 口径校验，不合法候选只能作为中间候选被拒绝，不得进入 validated best。
- 汇总 TRA-FixGurobi runtime 与 Gurobi baseline runtime，并检查 TRA runtime 是否至少快 20%。
- 解释 `STACK-S3` 和 `STACK-S4` 中 TRA fixed `XYZU` 可行解优于现有 Gurobi baseline 的原因，区分 baseline 参数/剪枝/时间限制/最优性证明问题与真实建模差异。
- 对比 Gurobi baseline 解、TRA 当前 best 解和 fixed global 注入解的结构差异，识别 X/Y/Z/U 层的关键改进模式。
- 基于结构差异新增或增强 TRA-FixGurobi 搜索算子，让 TRA 更快找到与 Gurobi 最优一致或经 global 证明合法的 best 解。

## Impact
- Affected specs: TRA-FixGurobi 验收、global fixed `XYZU` 注入验证、S1-S9 baseline 对比、搜索候选合法性审计。
- Affected code: `Gurobi/tra.py`、`Gurobi/tra_gurobi.py`、`Gurobi/resource_time_alns/fixgurobi_evaluator.py`、`Gurobi/resource_time_alns/engine.py`、`experiments/run_fixgurobi_replay.py`、实验结果汇总脚本或命令。
- 不允许影响：`Gurobi/global_xyzu.py` baseline 约束语义。

## ADDED Requirements
### Requirement: S1-S9 批量验收
系统 SHALL 支持对 `STACK-S1` 到 `STACK-S9` 批量运行 TRA-FixGurobi，并生成结构化汇总，包含 TRA Cmax、global 注入 Cmax、Gurobi baseline Cmax、TRA runtime、Gurobi runtime、speedup、注入状态和失败原因。

#### Scenario: 所有 case 完成汇总
- **WHEN** 用户运行 S1-S9 TRA-FixGurobi 验收流程
- **THEN** 系统输出每个 case 的 Cmax 对齐状态、runtime 对比和是否满足验收要求

### Requirement: Cmax 与 global 注入一致
系统 SHALL 以 global fixed `XYZU` 注入结果作为 TRA 输出解合法性的权威判断，并要求 TRA native/replay Cmax 与 global 注入 Cmax 一致。

#### Scenario: TRA Cmax 与 global 注入一致
- **WHEN** TRA 输出一个 S1-S9 case 的完整初始解或 best 解
- **THEN** fixed `XYZU` 注入 global Gurobi 返回可行状态，且 global Cmax 与 TRA Cmax 差异在容差内

### Requirement: 比 baseline 更小的 Cmax 必须注入校验
系统 SHALL 对任何 `TRA Cmax < Gurobi baseline Cmax` 的 case 自动执行 global 注入校验，并输出校验结论。

#### Scenario: TRA Cmax 小于 Gurobi baseline
- **WHEN** 某个 case 的 TRA Cmax 小于对应 Gurobi baseline Cmax
- **THEN** 系统执行 fixed `XYZU` 注入 global Gurobi
- **THEN** 若注入不可行或 Cmax 不一致，系统标记该 TRA 解违反 global 约束
- **THEN** 若注入可行且 Cmax 一致，系统标记需要复核 baseline 参数、剪枝参数、最优性证明和运行配置一致性

### Requirement: 搜索过程中排除违反约束的解
系统 SHALL 保证 TRA 搜索过程中违反 global 约束的候选不会进入 validated best，并记录被拒绝原因。

#### Scenario: 候选违反 global 约束
- **WHEN** TRA 搜索产生不可注入 global 的候选解
- **THEN** 该候选被标记为 infeasible、hard reject 或 `F_raw=inf`
- **THEN** 搜索不会用该候选更新 `best_validated` 或最终导出 best

### Requirement: runtime 至少快 20%
系统 SHALL 对比 TRA-FixGurobi runtime 与 Gurobi baseline runtime，并判定是否满足 `TRA runtime <= 0.8 * Gurobi runtime`。

#### Scenario: runtime 验收通过
- **WHEN** S1-S9 的 TRA-FixGurobi 与 Gurobi baseline 结果均已生成
- **THEN** 系统对每个 case 输出 speedup，并标记是否满足快 20% 的要求

### Requirement: 解释 S3/S4 baseline 差异
系统 SHALL 对 `STACK-S3` 和 `STACK-S4` 中 TRA fixed `XYZU` 注入可行且 Cmax 小于 baseline 的现象进行复核，明确 Gurobi baseline 未找到该解的原因。

#### Scenario: TRA 可行解优于 baseline
- **WHEN** TRA 解 fixed `XYZU` 注入 global 可行且 Cmax 小于 baseline Cmax
- **THEN** 系统复核 baseline 使用的 runtime config、剪枝参数、MIP gap、time limit、best bound 和求解状态
- **THEN** 系统输出该差异属于 baseline 未证明最优、参数口径不一致、剪枝排除了该解，还是存在其他建模/注入口径差异

### Requirement: 基于最优结构增强搜索算子
系统 SHALL 对比 Gurobi baseline 解与 TRA best 解的结构，识别能降低 Cmax 的可迁移模式，并将其转化为 TRA-FixGurobi 搜索算子或候选生成策略。

#### Scenario: 结构差异转化为算子
- **WHEN** S1-S9 中存在 TRA 未能快速达到目标 Cmax 或 runtime 不达标的 case
- **THEN** 系统比较 X/Y/Z/U 决策差异，包括订单拆分、站点排序、堆垛/命中 tote、机器人路径与 station rank
- **THEN** 系统新增或增强候选算子，使搜索优先尝试与已验证优质结构一致的局部变换

### Requirement: TRA 最优解与 Gurobi 一致且快 20%
系统 SHALL 以 global Gurobi 注入验证为准，要求 TRA 最终 best 与 Gurobi 认可的最优 Cmax 一致，并且 runtime 至少快 20%。

#### Scenario: 最终验收通过
- **WHEN** TRA-FixGurobi 在 S1-S9 上完成增强搜索
- **THEN** 每个 case 的 TRA best fixed `XYZU` 注入 global 可行
- **THEN** 每个 case 的 TRA Cmax 与 Gurobi 认可的目标 Cmax 一致
- **THEN** 每个 case 的 `TRA runtime <= 0.8 * Gurobi runtime`

## MODIFIED Requirements
### Requirement: TRA 结果验收口径
TRA 的验收口径 SHALL 从单纯 native Cmax 或内部 replay 结果，修改为必须同时满足 global fixed `XYZU` 注入可行、native/global Cmax 一致、runtime 对比达标。

### Requirement: TRA 搜索增强验收口径
TRA 搜索增强 SHALL 不仅保证合法性，还必须以 Gurobi baseline/复核最优结构为参照，证明增强算子能更快收敛到 global Gurobi 认可的目标 Cmax。

## REMOVED Requirements
### Requirement: 仅凭 TRA native Cmax 判断优劣
**Reason**: native Cmax 可能因回放口径或候选合法性问题与 global Gurobi 约束不一致，不能作为唯一合法性依据。
**Migration**: 使用 global fixed `XYZU` 注入 Cmax 作为权威合法性校验，并保留 native Cmax 作为一致性检查字段。
