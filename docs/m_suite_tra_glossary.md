# M1-M9 TRA 术语表

| 术语 | 定义 |
| --- | --- |
| Gurobi 基线 | 飞书文档 4.2 当前 M1-M9 结果及其对应实例、约束、剪枝和求解器配置。 |
| TRA-Gurobi | 按论文三组核心变量轮转；内层以 Gurobi 求无站台等待的路线松弛邻域，外层以原始 Gurobi 释放路线、队列与时间并认证候选。 |
| TRA-Fast | 使用论文 `Com_Proc/DP` 同构的低成本候选评价，并减少外层精确认证频率；最终仍须通过统一 verifier。 |
| 稀疏精确校准 | TRA-Fast 在下界或代理不确定度触发时，于轮转外层调用短时 Gurobi 精确子问题；调用时间计入 runtime。 |
| 双层轮转 | 内层松弛筛选与外层完整模型校准交替进行，并轮转固定变量组合。 |
| `X_group` | 论文 `psi` 在当前模型中的投影：以 `x[work_unit,slot]` 为主赋值，`a/sku_use` 由约束派生。 |
| `S_visit` | 论文 `vartheta` 的投影：每个 `(slot,stack)` 一次赋值到合法 station 或 `inactive`，并链接到 `pair_activate` 与 Y/Z。 |
| `R_assign` | 论文 `alpha` 的投影：每个 slot 一次赋值到 robot 或 `inactive`，其中 active slot 必须选择 robot；M1-M9 以 `slot_robot` 为主载体，`passX/route_owner` 派生。 |
| `inactive` 哨兵 | 为未使用的 slot 或 slot-stack 提供确定的一次赋值，使固定核心块和 Hamming 距离在激活状态变化时仍有完整定义；不代表物理资源。 |
| `Y_station` | `Y` 中只表示 slot 选择哪个 station 的投影，属于外层固定结构。 |
| `Y_rank` | `Y` 中表示 station 内服务队列位置的投影，对应论文外层释放的 `sigma` 语义。 |
| 无等待路线松弛 | 保留路线弧、pickup-delivery 先后和处理时间，但移除 station 独占队列及等待递推；对应 Appendix D 的 `M_Batch^R`。 |
| 结构投影 | 外层固定的 `X_group + Y_station + Z_inventory + S_visit + R_assign`；不含 route arc、station rank 或连续时间。 |
| incumbent canonicalization | 只在签名完全等价的 slot、tote、station、robot 轨道内确定性重编号，使固定投影满足共享 symmetry；物理任务与 Cmax 不变。 |
| safe prune | 有数学证据表明不会删除任何可行最优解的剪枝。 |
| heuristic prune | top-k、KNN、hard cap 等依赖排序或经验阈值、无法单独保证最优性保持的剪枝。 |
| pruning manifest | 一次运行实际使用的候选实体、inventory action、route node/arc、保护集合、数值界、目标、约束与对称性指纹的规范化只读快照。 |
| master domain | 由共享预处理器根据当前实例和 canonical warm-start 一次性生成、供 Gurobi、TRA-Gurobi 和 TRA-Fast 共同使用的唯一可行域。算法层不得修改它。 |
| 实例指纹 | 对需求、库存、地图、资源、随机种子和关键配置规范化后计算的稳定标识。 |
| incumbent provenance | 当前最好可行解的来源记录，例如自然搜索、warm-start、结构引导或最终校准。 |
| required replay | 依赖历史结构或 verified export 直接重放解的兜底路径；正式验收禁止。 |
| structure guidance | 把历史 Gurobi 解结构作为提示或初解；正式验收禁止。 |
| canonical warm-start | 共享预处理器仅根据当前实例和固定 seed 生成的统一初解；其 stack 与 route arc 必须进入保护集合，不读取历史 Gurobi 解。 |
| 算法 warm-start | master domain 冻结后由算法生成或改进的初解；可以影响搜索速度，但不得再改变 master domain。 |
| protected warm domain | canonical warm 使用的实体必须保留在共享搜索域中，但并不固定为最终选择；任何计时前 Gurobi 修复或启发式改进均被禁止。 |
| 一致性 | 三种算法在同一实例与约束口径下，由统一 verifier 重算得到相同 Cmax。 |
| runtime | 正式验收的搜索阶段墙钟时间：从第一轮双层轮转开始，到轮转搜索终止。轮转内部精确子问题、校准和修复计入；轮转前预处理与轮转后外部校验/导出单独报告。 |
| 20% 加速 | candidate runtime <= 0.8 * baseline runtime。边界值按未舍入秒数判定。 |
| 正式运行 | 每个 case 单次运行；Cmax 由最终统一 verifier 验收，runtime 按轮转日志中的 time-to-target 验收。 |
| target-blind | 搜索过程不知道 4.2 Cmax，不使用目标引导、目标探针、目标 cutoff 或按目标提前停止。 |
| time-to-target | 从第一轮双层轮转开始，到日志中首次出现并通过内部可行性检查的 4.2 Cmax 解的累计搜索时间；运行后由验收器计算。 |
| sanitized domain policy | 从历史 summary 白名单抽取的有效剪枝、对称性和模型配置；不含 Cmax、目标值、解结构、warm 派生实体或 export 路径。 |
| feasible solution event | 完整 outer 的 solver-feasible incumbent 在正式计时内留下的带时间戳快照；通过内部只读 verifier 后才可供事后 time-to-target 回放。 |
| 基准复合目标 | 旧 `GlobalXYZU` 实际使用的完整目标表达式。TRA-Gurobi 外层和候选接受严格沿用该表达式；`Cmax` 单独记录并用于外部验收，不在正式路径中改成词典序第一目标。 |
| 未决外层候选 | 外层限时结束但 `ObjBound` 尚未证明当前结构不能改进的候选。它进入保留预算重试，不能被记为 infeasible 或已证明无改进。 |
| `F_relax` | 基准复合目标在无站台等待内层模型上的可证明下界目标；只有它及其 `ObjBound` 可用于证明性剪枝。 |
| `repair_risk` | 站台拥堵、等待冲突、余量和 warm 扰动等非证明性候选风险，只用于排序，不得用于硬剪枝或正式接受。 |
| 自然精英池 | inner MIP 在正常求解过程中由 callback 收集的至多 8 个互异结构候选；不额外开启 solution-pool 搜索，最终只选一个进入外层。 |
| 只读 verifier | 正式计时结束后仅检查已有解的覆盖、资源、路径、时序和 Cmax，不执行任何修复或优化。 |
