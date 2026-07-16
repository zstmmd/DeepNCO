# ADR 0001: M1-M9 TRA 验收与双层轮转契约

- 状态：Proposed
- 日期：2026-07-15
- 范围：M1-M9 的 TRA-Gurobi 与 TRA-Fast

## 背景

飞书文档 4.2 的中规模结果是当前 Gurobi 基线。现有无结构验收中，M1 已通过，M2 因候选 stack 与 warm-start 口径漂移而无法在预算内复现基线 Cmax。变量数和约束数接近并不等价于候选集合、对称性、剪枝图和初解一致。

参考论文采用双层轮转算法：内层求解松弛模型，外层求解原模型；三组核心变量按两两组合轮转；下界门控用于跳过无潜力组合。本文实现保留双层轮转结构，但把不可审计的启发式筛除替换为可验证的剪枝契约。

## 决策

1. 保留双层轮转：内层负责低成本邻域筛选和下界估计，外层负责完整约束下的精确校准与最终验收。
2. Gurobi、TRA-Gurobi、TRA-Fast 必须先调用同一个确定性共享预处理器，从当前实例生成一份 canonical warm-start，再据此生成唯一的 master domain 和 pruning manifest。canonical warm-start 使用的 stack 与 route arc 必须受保护。三种算法只能在该 master domain 上搜索，不得各自重建或修改候选集合。manifest 至少包含候选 stack/station、route arc 集、受保护 arc、slot 上界、对称性开关、MIP gap、线程数、seed 和 canonical warm-start 指纹。
3. 候选剪枝分为两类：
   - safe prune：由支配关系、容量、时间窗、负载区间或可证明下界排除，允许用于正式验收；
   - heuristic rank/cap：top-k、KNN、hard cap 等无法证明不损失最优解的筛除，允许保留旧 Gurobi 的确定性策略，但只能由共享预处理器一次性确定并在 manifest 中单独标记。算法层不得二次剪枝。
4. TRA-Gurobi 的最终 Cmax 必须等于 Gurobi 基线，runtime 至少降低 20%。TRA-Fast 的最终 Cmax 也必须等于 Gurobi 基线，runtime 至少比 TRA-Gurobi 降低 20%。低于基线同样判失败，因为它表示实例或约束口径漂移。
5. 最终 Cmax 必须由统一 verifier 根据任务、路径和时序重算，不能只读取 solver objective 或沿用 structure export 的已验证值。
6. M1-M9 按 M1 -> M9 顺序闸门推进。任一 case 的一致性失败时，暂停更大 case 的正式验收，先生成 manifest diff 和首个差异证据。
7. 正式 runtime 采用搜索阶段墙钟时间，以匹配飞书 4.2 的 `gurobi_runtime_sec=model.Runtime` 口径。TRA-Gurobi 与 TRA-Fast 均在 canonical warm-start 和 master domain 已就绪后、第一轮双层轮转开始前启动计时，在轮转搜索终止时停止。轮转内部的精确子问题、校准和修复必须计入；轮转前预处理以及轮转后的外部 verifier 和导出不计入正式 runtime，但必须分别报告。
8. 正式验收禁止读取或重放 Gurobi 历史解、structure export、verified export 和 required replay。TRA 仅可使用由当前实例和当前算法独立生成的 warm-start；历史基线只提供验收用 Cmax、runtime 与冻结配置，不向搜索过程暴露解结构。
9. canonical warm-start 参与共享 master domain 的一次性构造，其 stack 与 route arc 必须保护，以保证初解在剪枝后仍可行。master domain 冻结后，各算法后续生成或改进的 warm-start 只能设置初始变量值或初始上界，不得再扩张或缩减候选 stack/station、route arc 或 slot 上界，也不得生成算法专属保护边。
10. TRA-Gurobi 和 TRA-Fast 的搜索过程不得读取飞书 4.2 Cmax，不得使用 known-target guidance、target probe、target-based cutoff、`BestObjStop` 或按目标值提前终止。4.2 Cmax 仅由外部验收器读取。正式 runtime 从迭代日志中事后计算，为第一轮双层轮转开始至首次产生并通过内部可行性检查的目标 Cmax 解的累计搜索时间。
11. TRA-Fast 允许在双层轮转外层调用短时 Gurobi 精确子问题，调用时间计入正式搜索 runtime。TRA-Gurobi 采用高频精确评估；TRA-Fast 仅在下界潜力充足、代理不确定度高或 incumbent 需要精确确认时触发稀疏校准。两者必须使用同一 master domain。
12. 双层轮转结束或正式计时停止后，禁止调用任何继续优化的 Gurobi 修复、校准、structure replay 或其他补解路径。外部 verifier 只能读取并验证计时内已经产生的 incumbent；计时后得到的更优解不得用于验收。

## Gurobi 基线

下表是飞书 4.2 的正式历史基线。TRA-Gurobi 的 20% runtime 门槛继续按该表计算。共享 master domain 上会重跑 Gurobi，但该次运行用于验证旧 Cmax 可复现、产出同域 manifest 和记录诊断，不替换 4.2 runtime 门槛。

| Case | Cmax | Runtime (s) | TRA-Gurobi 上限 (s) | TRA-Fast 上限 |
| --- | ---: | ---: | ---: | ---: |
| M1 | 582 | 360.18 | 288.14 | TRA-Gurobi 的 80% |
| M2 | 805 | 384.04 | 307.23 | TRA-Gurobi 的 80% |
| M3 | 830 | 663.53 | 530.82 | TRA-Gurobi 的 80% |
| M4 | 1098 | 567.06 | 453.65 | TRA-Gurobi 的 80% |
| M5 | 863 | 663.67 | 530.94 | TRA-Gurobi 的 80% |
| M6 | 1064 | 1039.62 | 831.70 | TRA-Gurobi 的 80% |
| M7 | 1538 | 1314.21 | 1051.37 | TRA-Gurobi 的 80% |
| M8 | 1411 | 1837.86 | 1470.29 | TRA-Gurobi 的 80% |
| M9 | 2110 | 2608.73 | 2086.98 | TRA-Gurobi 的 80% |

## 论文之上的改进点

1. 用可重放的 pruning manifest 取代仅由阈值控制的隐式搜索空间。
2. 用下界证书门控外层求解；阈值只决定是否提前投入预算，不得直接证明最优性。
3. 用 incumbent provenance 记录每个可行解来自自然搜索、warm-start、结构引导还是外层校准，禁止把结构导入的历史解记为自然求解结果。
4. 用预算感知轮转替代固定三轮停机：根据下界潜力、最近改进率和外层校准成本动态分配层预算，同时保留无改进终止条件。
5. 用统一 verifier 和 manifest diff 把“Cmax 相同”提升为可复现实验结论。
6. 将 canonical warm-start 保护集中到共享预处理阶段，并在 manifest 中显式记录；避免算法专属 warm-start 造成三套不同搜索空间。

## 待确认

1. 若 Gurobi 域审计在共享 master domain 上不能复现飞书 4.2 历史 Cmax，该 case 是否直接判定 domain audit 失败并停止 TRA 正式验收？

## 已确认

- runtime 从第一轮双层轮转开始计时；canonical warm-start/master domain 构造和外部校验/导出单独报告，不计入 20% 门槛。轮转内部的精确求解、校准和修复计入。
- 正式验收采用无历史结构的自然搜索；禁止 structure export、verified export 和 required replay，warm-start 必须由 TRA 当前运行自行生成。
- 每个 case 以单次正式运行判定 Cmax 与 runtime，不要求重复运行中位数。
- 采用由当前实例的 canonical warm-start 统一保护 stack/arc 后生成的共享 master domain；各算法不得用自己的后续 warm-start 改写 domain。20% 门槛仍使用飞书 4.2 归档 runtime。
- 共享 master domain 允许沿用旧 Gurobi 的确定性 top-k/KNN/hard cap；这些启发式剪枝只在共享预处理阶段执行一次，三种算法共用相同结果。
- 搜索过程禁止读取 4.2 Cmax；正式 runtime 由日志事后回放得到首次命中目标 Cmax 的累计轮转时间。
- TRA-Fast 允许轮转内短时 Gurobi 精确校准，且计入 runtime；其加速来自稀疏触发，不来自放松可行域或省略最终可行性要求。
- 计时结束后禁止补解；统一 verifier 只能验证计时内已有 incumbent。

## 后果

这套契约会先增加诊断与指纹生成成本，但能直接定位 M2 当前的候选集合和 warm-start 漂移。只有 manifest 一致后，runtime 对比才具有论文可辩护性。
