# M1-M9 TRA 术语表

| 术语 | 定义 |
| --- | --- |
| Gurobi 基线 | 飞书文档 4.2 当前 M1-M9 结果及其对应实例、约束、剪枝和求解器配置。 |
| TRA-Gurobi | 保留双层轮转搜索，并以 Gurobi 精确子问题完成候选评估、修复或外层校准的算法。 |
| TRA-Fast | 使用低成本代理评估和较少精确校准的双层轮转算法；最终仍须通过统一 verifier。 |
| 稀疏精确校准 | TRA-Fast 在下界或代理不确定度触发时，于轮转外层调用短时 Gurobi 精确子问题；调用时间计入 runtime。 |
| 双层轮转 | 内层松弛筛选与外层完整模型校准交替进行，并轮转固定变量组合。 |
| safe prune | 有数学证据表明不会删除任何可行最优解的剪枝。 |
| heuristic prune | top-k、KNN、hard cap 等依赖排序或经验阈值、无法单独保证最优性保持的剪枝。 |
| pruning manifest | 一次运行实际使用的候选集合、route arc、保护集合、上下界、对称性和求解参数的规范化快照。 |
| master domain | 由共享预处理器根据当前实例和 canonical warm-start 一次性生成、供 Gurobi、TRA-Gurobi 和 TRA-Fast 共同使用的唯一可行域。算法层不得修改它。 |
| 实例指纹 | 对需求、库存、地图、资源、随机种子和关键配置规范化后计算的稳定标识。 |
| incumbent provenance | 当前最好可行解的来源记录，例如自然搜索、warm-start、结构引导或最终校准。 |
| required replay | 依赖历史结构或 verified export 直接重放解的兜底路径；正式验收禁止。 |
| structure guidance | 把历史 Gurobi 解结构作为提示或初解；正式验收禁止。 |
| canonical warm-start | 共享预处理器仅根据当前实例和固定 seed 生成的统一初解；其 stack 与 route arc 必须进入保护集合，不读取历史 Gurobi 解。 |
| 算法 warm-start | master domain 冻结后由算法生成或改进的初解；可以影响搜索速度，但不得再改变 master domain。 |
| 一致性 | 三种算法在同一实例与约束口径下，由统一 verifier 重算得到相同 Cmax。 |
| runtime | 正式验收的搜索阶段墙钟时间：从第一轮双层轮转开始，到轮转搜索终止。轮转内部精确子问题、校准和修复计入；轮转前预处理与轮转后外部校验/导出单独报告。 |
| 20% 加速 | candidate runtime <= 0.8 * baseline runtime。边界值按未舍入秒数判定。 |
| 正式运行 | 每个 case 单次运行；Cmax 由最终统一 verifier 验收，runtime 按轮转日志中的 time-to-target 验收。 |
| target-blind | 搜索过程不知道 4.2 Cmax，不使用目标引导、目标探针、目标 cutoff 或按目标提前停止。 |
| time-to-target | 从第一轮双层轮转开始，到日志中首次出现并通过内部可行性检查的 4.2 Cmax 解的累计搜索时间；运行后由验收器计算。 |
| 只读 verifier | 正式计时结束后仅检查已有解的覆盖、资源、路径、时序和 Cmax，不执行任何修复或优化。 |
