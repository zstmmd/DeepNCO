# ADR 0002：论文式双层轮转 TRA-Gurobi

- 状态：已接受，待实施
- 日期：2026-07-18
- 关联：ADR 0001 的验收与目标盲测契约

## 目标

在 M1-M9 上按论文的 Two-Layer Revolving Algorithm（TRA）实现
TRA-Gurobi，同时保持与基准 Gurobi 相同的有效搜索域。旧 4.2 的
`Cmax` 和 `runtime` 只用于运行后的外部验收，不进入求解器、轮转决策、
停止条件或参数选择。

## 论文结构

论文的每个轮转过程包含两次求解：

1. 内层求解松弛模型 `M_Batch^R`。三个核心变量中固定两个，释放一个。
2. 外层求解原模型 `M_Batch`。固定九个结构变量，重新优化其余四个
   路线顺序与时间变量。

三个固定组合依次轮转。仅当对应下界仍有改进空间时执行过程；改进后
更新 incumbent 并重置无改进计数。

补充材料 Appendix D/E 进一步明确：

- `M_Batch^R` 并未删除 pod 路线。它保留 `delta` 路线与 `phi` 到达时间，
  但删除站台等待和站内 pod 排序相关的 `rho/sigma` 约束；其到达递推使用
  不含站台等待的 (D-2)。
- 求得无等待到达时间后，以 (D-3) 作为后续含等待调度的到达下界。
- Sub-algorithm 1 固定 `alpha + psi`，变动 `vartheta`；Sub-algorithm 2
  固定 `alpha + vartheta`，变动 `psi`；Sub-algorithm 3 固定
  `vartheta + psi`，变动 `alpha`。
- 论文不是在每个邻域内直接求完整 MIP。`Com_Proc` 用 DP1 求每个 pod 的
  无等待 TSP 路线，用 DP2 加入处理时间，用 DP3 恢复站台等待和 pod 服务
  顺序并快速评价候选。
- 每个 procedure 最终再求原模型；固定的九个结构变量为
  `kappa/alpha/vartheta/beta/psi/xi/theta/mu/gamma`，释放的四组变量为
  `delta/phi/rho/sigma`。

## 当前模型的候选投影

论文变量不能直接等同于现有整块 `X/Y/Z/U`。建议使用以下语义块：

| 论文核心语义 | 当前模型投影 | 说明 |
| --- | --- | --- |
| 订单分组 `psi` | `X_group`：以 `x` 为主赋值，`a/sku_use` 由约束派生 | 决定订单工作单元进入哪个 slot |
| pod/stack 访问站台 `vartheta` | `S_visit`：`(slot, stack)` 到 `station/inactive` 的一次赋值，投影到 `pair_activate` 与 Y/Z | 表示 stack-slot-station 的访问关系 |
| 机器人承担 pod/task `alpha` | `R_assign`：每个 slot 到 `robot/inactive` 的一次赋值，投影到 `slot_robot/passX/route_owner` | active slot 必须选 robot；不包含路线顺序和连续时间 |

外层固定的结构壳候选为 `X_group + Y_station + Z_inventory + S_visit +
R_assign`。当前 `Y` 必须拆成只表示站台选择的 `Y_station` 与表示站台队列
位置的 `Y_rank`。外层必须释放 `Y_rank`、`route_arc`、`route_time`、
`route_load`、`route_finish`、`arrival/start/finish` 和 `Cmax`，使原模型
重新决定路线顺序、站内队列和时间。

三个核心块均使用完整的一次赋值载体，以便固定和计算邻域距离：

- `X_group[u, slot]`：每个 work unit 恰选一个同订单 slot；
- `S_visit[(slot, stack), q]`：`q` 为某个合法 station 或 `inactive`；
- `R_assign[slot, q]`：`q` 为某个 robot 或 `inactive`。M1-M9 的
  `u_same_slot_same_robot=true`，因此以 `slot_robot` 为最小非冗余载体；
  `passX/route_owner` 只由它和 active route task 派生。

`inactive` 使未使用 stack/slot 也有确定值，避免固定另外两个核心块时，新激活
任务因“上一解没有机器人值”而被错误禁止。它是投影视图，不新增物理选择；
链接约束保证投影与原 `a/pair_activate/passX` 完全等价。

## 论文邻域在当前模型中的对应

| Procedure | 固定核心块 | 释放核心块 | 论文邻域 | 当前模型候选邻域 |
| --- | --- | --- | --- | --- |
| F1 | `R_assign + X_group` | `S_visit` | pod 增加 station；两个 pod 交换 station 集 | visit 重定向/增加；两个 active stack/task 交换 station 投影 |
| F2 | `R_assign + S_visit` | `X_group` | 两个 cluster 交换 order | 同一 order 内两个 slot 交换或 relocate 工作单元，并保持容量与覆盖 |
| F3 | `S_visit + X_group` | `R_assign` | 两个 pod 交换 robot | active slot/task 的 robot relocate 或 swap，并保持归属一致性 |

论文的 order cluster 与当前 slot 不是同一对象，因此 F2 只能保持其“交换
分组成员”的语义，不能机械照搬跨订单交换。论文的一台机器人对应一个 pod，
当前机器人可执行多个 pickup-delivery task，因此 F3 还需要机器人工作量和
任务归属一致性修复。

## 适配后的内层松弛

TRA-Gurobi 的内层应是论文 `M_Batch^R` 的当前模型投影，而不是完整
`GlobalXYZU`，也不是完全删除路线：

1. 保留三类结构变量、inventory 可行性、机器人任务归属、共享主域内的
   route task/arc，以及 pickup-delivery 先后关系。
2. 固定两个核心块，只对第三个核心块开放论文对应的 local-branching 邻域。
3. 保留无等待路线到达递推与处理时间，删除 `Y_rank`、站台独占队列、
   `start >= previous finish` 等等待耦合。
4. 内层结果中的路线弧只用于得到候选结构和无等待下界，不在外层固定。
5. 外层用原始 `GlobalXYZU` 固定结构投影，释放路线、rank 和时间，只有外层
   返回内部可行解时才能更新正式 incumbent。

这比论文的 `Com_Proc + DP1-DP3` 更适合 TRA-Gurobi：保留论文的合法松弛
边界，但用可缓存的 Gurobi 松弛模型求邻域最优；TRA-Fast 再承担
`Com_Proc/DP` 式快速评价与稀疏外层认证。

## 与当前实现的差异

1. 当前 `U` 同时包含任务归属、路线弧和连续时间，粒度过粗。
2. 当前 `Y` 同时包含 station 与 rank，无法做到论文要求的“固定 station、
   释放站内服务顺序”。
3. 当前 revolving scope 主要在 `X/Y/Z` 间互补固定，并未把 `U` 的归属
   部分作为论文的第三个核心变量。
4. 正式路径把 `XYZ` 临时映射为 `GLOBAL`，并先执行 canonical seed outer；
   这是排障措施，不是论文中的三过程轮转。
5. 当前候选修复可能先花完时间预算，尚未形成“松弛内层产生结构候选，
   原模型外层认证”的稳定边界。

## 剪枝对齐契约

参数相同不足以证明搜索域相同。正式运行需先构造 schema v3 共享主域 manifest，
并冻结剪枝后的有效集合，而不是历史解：

- 完整实例与物理常量指纹：逐 SKU 需求、tote 层序/库存、地图坐标、station、
  robot、容量、速度和处理时间；
- slot/work-unit 集合、`x` keys、active-slot 上下界与 tight time bounds；
- 每个订单的候选 stack，以及 flip/sort/carry/hit/noise/flip-hit 全部变量 keys 和
  sort interval 描述；
- `Y_station` keys、允许 rank/max-rank，以及 `(slot, stack, station)` route-task
  集合；
- 带语义 key 的 route node、travel time、可用 route arc、robot eligibility 和
  warm-protected arc；
- route-node time UB、arc Big-M、load bound 等所有会缩小显式变量域的数值界；
- objective expression/coefficient、基础变量/约束、有效不等式、对称破除的
  分组件 fingerprint；
- 每条剪枝规则的 `safe/heuristic` 标签、配置来源、before/after count 和结果
  集合 hash；
- 当前实例、sanitized case policy、canonical warm、各域分区和整份 manifest
  的 SHA-256。

基准 Gurobi、TRA 内层和 TRA 外层必须读取同一 manifest，禁止各自重新执行
top-k、KNN、warm cap、dominance、time-window/load prune 或任何候选排序。
consumer 只能从 manifest 枚举变量 keys 和数值界；当前 `_prepare()` 中“先重算
剪枝、最后再覆盖集合”的方式不满足正式契约，必须新增 fail-closed 的 manifest
consumer 路径。Gurobi presolve、cuts、heuristics 等仅改变搜索过程的参数不
属于主域，但任何显式删变量、删弧、缩界操作都属于主域。

canonical warm 的 stack 和可行路线弧在主域生成时受保护；主域冻结后，
warm 只作为 MIP start，不再隐式增删候选。

## 当前 Gurobi 剪枝审计

历史 `gurobi_summary.json` 中的 `config + diagnostics` 显示以下实际生效值。
验收 helper 从旧配置文件回填的 M4-M9 值与之不一致，因此不得把 helper
回填参数直接当作主域来源。

| Case | station top-k | base route prune | time-window prune | load prune | pickup KNN | tight slot UB |
| --- | ---: | --- | --- | --- | ---: | --- |
| M1 | 999 | on | off | on | 0 | on |
| M2 | 999 | on | off | on | 0 | on |
| M3 | 1 | on | off | on | 0 | on |
| M4 | 1 | on | off | on | 0 | on |
| M5 | 1 | on | off | on | 0 | on |
| M6 | 2 | off | off | on | 0 | on |
| M7 | 1 | on | on | on | 5 | on |
| M8 | 1 | on | off | on | 5 | on |
| M9 | 1 | on | off | on | 5 | on |

所有案例的 `candidate_stack_topk=999`、`max_candidate_stacks_per_order=0`、
`enable_warm_candidate_stack_prune=false`。但当前候选生成仍无条件执行 stack
dominance；station top-k、tight slot UB、M7-M9 的 KNN，以及依赖 warm 上界的
time-window 剪弧都可能改变候选域，不能称为已证明的 safe prune。

剪枝按三类管理：

1. **结构安全剪枝**：非法 start/end 弧、pickup-delivery 先后矛盾、容量上
   必不可能的弧、合法有效不等式。记录规则和证明标签。
2. **启发式域限制**：top-k、KNN、warm-derived slot/time bound、未证明的
   dominance。必须记录最终保留的实体集合及 `heuristic` 标签。
3. **搜索参数**：Gurobi presolve、cuts、MIPFocus、Heuristics 等，不删除
   显式变量时只记录参数，不作为 master-domain 实体集合。

正式对齐的技术判据是 Gurobi、TRA 内层、TRA 外层读取的 slot、stack、
station-task、inventory-action、route-arc 集合哈希完全相同，而不是参数文本
看起来相同。历史 summary 用于还原配置口径；正式 manifest 仍由当前实例与
canonical warm 生成，禁止读取历史解结构。

历史 summary 不能由正式求解进程直接读取。单独的 sanitizer 只白名单导出
effective pruning/symmetry/model 配置，删除 objective value、Cmax、bound、
solution、warm-derived 实体值和 export 路径。M1 缺失的
`sort_hit_tote_threshold` 不能填当前默认值：M1 summary 同时缺少该配置、诊断值
和 `sort_hit_tote_threshold_count`，对应旧模型没有 `SortByHitThreshold*` 约束；
case policy 应显式写为 `enable_sort_hit_tote_threshold=false` 并记录 legacy-code
provenance。M2-M9 则为 enabled、threshold 3。不得从历史解反推。正式 runtime
budget 另存为只含 case/runtime 的文件；验收 target 再单独存放并只供事后进程
读取。

## 待决策

剩余实现细节按本 ADR 的确定性默认值执行；如代码审计发现会改变可行域、正式
目标或计时口径的新问题，再单独提升为 ADR 决策。

## 决策记录

### 2026-07-18：第三核心块采用机器人任务归属

采用 `R_assign` 作为论文 `alpha` 的对应核心块。现有 `U` 拆为：

- `R_assign`：以 `slot_robot`（或无同-slot约束时的 task-robot 一次赋值）为主
  载体；`passX/route_owner` 是其派生投影，不重复固定或计距；
- `R_sequence`：`route_arc`；
- `R_time`：`route_time`、`route_load`、`route_finish` 及关联连续时间变量。

只有 `R_assign` 参与三块轮转。`R_sequence` 与 `R_time` 在外层原模型中
始终释放并重新优化，不从上一解做精确固定。

### 2026-07-18：补充材料纠正内外层变量边界

此前“内层不含离散路线弧”提案作废。Appendix D 表明论文松弛模型保留
`delta/phi`，删除的是站台等待与服务顺序耦合。当前模型据此采用：

- 内层保留共享主域内的 `route_arc` 和无等待 route time；
- 外层仍不固定 `route_arc`；
- `Y_station` 属于固定结构投影，`Y_rank` 属于外层释放的队列变量。

### 2026-07-18：TRA-Gurobi 采用松弛 MIP 与 local branching

TRA-Gurobi 保留论文 F1/F2/F3 的三种邻域语义。每次 procedure：

1. 在另外两个核心块上施加 incumbent 等值固定；
2. 对释放核心块施加论文邻域对应的 local-branching 约束；
3. 使用可缓存的无站台等待路线 MIP 求内层候选；
4. 将内层结果投影为九类结构决定；
5. 使用完整 `GlobalXYZU` 固定结构投影，释放 route/rank/time 后求外层认证。

TRA-Fast 后续采用论文 `Com_Proc/DP` 同构的快速评价器，并以更稀疏的外层
Gurobi 认证与 TRA-Gurobi 区分。

### 2026-07-18：内层只固定两个核心块

每个 procedure 只把另外两个核心块固定为 incumbent 投影。释放核心块在
local-branching 邻域内变化；`kappa/beta/xi/theta/mu/gamma` 对应的当前模型
依赖结构变量，以及无等待路线变量，均允许 Gurobi 在共享主域内联动修复。

不得把上一解的完整 `Y/Z` 或路线作为隐式固定条件。这里的“自由”仅指
子问题内可优化，不允许越过 master-domain manifest 中的 slot、stack、
station、route task、route arc 和 inventory action 域。

### 2026-07-18：采用语义化自适应 VNS 邻域

local-branching 距离只统计释放核心块的主赋值指标，不统计由约束联动产生的
派生变量：

- `N1`：恰好一次可行 relocate，对 one-hot 赋值的原始 Hamming 距离为 2；
- `N2`：两个实体交换标签，保持相应标签计数，原始 Hamming 距离为 4；
- `N3`：最多四个一次赋值实体改变、等价于至多两个组合 move，原始 Hamming
  距离上限为 8，只在停滞时启用。

F1 对 `S_visit` 使用 relocate/swap；F2 对 `X_group` 以 swap 为主；F3 对
`R_assign` 使用 robot relocate/swap。获得改进后回到 `N1`，连续无改进才
逐级扩张。F2 的 N1 只允许在当前 active slot 之间 relocate，且源 slot 保持
非空；N2 明确保留 active-slot 集。邻域约束属于搜索策略，不写入
master-domain manifest。

### 2026-07-18：正式验收采用 baseline-aligned 共享域

正式 M1-M9 运行保留历史 Gurobi 实际启用的 top-k、KNN、tight slot UB、
route prune、dominance 等规则，但只允许共享预处理器执行一次。预处理器使用
当前实例、固定 seed 和 canonical warm 生成最终实体集合；Gurobi、TRA 内层、
TRA 外层随后只读同一个 manifest，并严格校验集合哈希。

该决定保证正式比较中的三条求解路径使用同一受限域，但不声称其中的启发式
限制保留了未剪枝原模型的全局最优解。报告必须将其称为
`baseline-aligned restricted domain`，并分别列出 safe/heuristic 标签。

历史解、历史 route sequence、历史 stack 选择不得用于生成正式 manifest。
历史 summary 仅用于恢复配置口径；缺失配置必须按旧代码语义显式补入 case
policy，禁止像当前 helper 一样从历史 solution export 反推
`sort_hit_tote_threshold`。字段缺失不自动等于当前 dataclass 默认值。

### 2026-07-18：投影对称约束并规范化 incumbent

完整外层保留每个 case 的基准 symmetry 配置。无等待内层只保留不依赖
`Y_rank`、station waiting 或完整 route time 的结构对称投影；删除的约束使
内层域更松，不会产生错误下界。

每轮开始前，对完全等价的 slot、tote、station 和 robot 轨道做确定性标签
规范化，使 incumbent 与固定投影满足共享对称约束。规范化必须通过等价签名
校验，不允许交换位置、距离、容量或库存签名不同的实体；规范化前后必须
验证物理任务集合和 `Cmax` 不变，并记录 permutation hash。

### 2026-07-18：证明性硬剪枝与 epsilon 延迟队列

为每个固定核心对计算论文三组下界的当前模型投影，并与 station workload、
robot assigned workload、order workload、task intrinsic arrival 等已证明有效的
下界取最大值。只有认证 `LB >= incumbent - tolerance` 时才永久剪枝。

论文相对差距门控改为调度规则：

- `(incumbent - LB) / incumbent > epsilon`：立即进入内层求解；
- `0 < (incumbent - LB) / incumbent <= epsilon`：进入延迟队列；
- 当前邻域完整停滞后，按潜在改进排序重访延迟候选。

限时 Gurobi 的 `ObjBound` 可作为认证下界；`ObjVal` 只是松弛模型可行解的
目标，不能用于硬剪枝。所有下界仅依赖当前实例、固定核心投影和 incumbent，
不得读取 4.2 Cmax。

### 2026-07-18：按旧 runtime 设置自适应硬预算

每例正式搜索墙钟预算为 `B_case = 0.8 * old_4.2_runtime`。预算只使用历史
runtime，不读取历史 Cmax。初始调度目标为约 30% 内层松弛、55% 外层认证、
15% N3/延迟队列保留；scheduler 可按剩余时间和实际 solve time 动态调整。

每个 procedure 最多把当前最优的一个内层结构候选送入外层。搜索在以下任一
条件满足时停止：达到 `B_case`；完成 50 次 procedure；延迟队列清空后连续
三个完整 F1-F2-F3 周期无内部可行改进。

canonical warm、master-domain 构造和内外层模型预编译在正式计时前完成；
第一轮 F1/F2/F3 开始时启动计时。轮转内的 MIP start 投影、求解、修复和内部
可行性检查全部计入预算。

### 2026-07-18：外层严格沿用旧 4.2 的复合目标

完整外层模型、outer incumbent 和候选接受均使用旧 `GlobalXYZU` 实际构造的
复合目标，而不是改为 `Cmax` 优先的词典序目标或纯 `Cmax` 目标。正式
master-domain manifest 必须保存目标表达式、逐项系数和变量域的指纹；所有
权重从该 case 的基准有效配置及实际模型构造结果恢复。配置中存在但旧代码未
加入目标表达式的罚项，不得在正式比较路径中补加。

外层候选只有在完整模型可行且复合目标按基准容差严格改进时，才替换当前
incumbent；较小的 `Cmax` 不能覆盖更差的基准复合目标。每个完整可行候选仍须
同时记录复合目标、verifier 重算的 `Cmax` 和首次出现时间。旧 4.2 `Cmax` 不
进入求解、候选接受或停止逻辑，只由运行后的验收器计算 time-to-target。

内层无等待松弛是候选生成器，其代理目标可以利用经证明的 `Cmax` 下界和局部
扰动代价排序，但不能直接更新正式 incumbent；正式目标改进只由完整外层确认。

### 2026-07-18：外层超时候选进入未决队列

每个候选的完整外层固定内层导出的结构投影，释放 route sequence、station
rank/waiting 和完整时间变量，并沿用基准复合目标。外层可加入仅依赖当前
incumbent 的严格改进 cutoff；该 cutoff 不读取旧 4.2 目标，也不改变可接受解
集合。

外层状态按证明能力处理：

- 找到完整可行且严格改进的解：立即按基准目标接受；
- `OPTIMAL`、`INFEASIBLE`，或 `ObjBound` 已证明 cutoff 下无解：淘汰该结构；
- 达到 `TIME_LIMIT` 且未找到改进解，但 `ObjBound` 仍允许改进：标记为
  `unresolved`，不得记为 infeasible 或普通 reject；
- `unresolved` 候选按认证下界与潜在改进排序，在完整 F1-F2-F3 周期停滞后从
  15% 保留预算中重试；每个结构最多进行一次保留预算重试，且不得突破
  `B_case`。

保留预算重试仍未给出改进解时，只能记录为 `budget_exhausted`，不能声称该结构
不可行或无改进空间。它可以在本次正式运行中停止参与调度，但必须在日志和
结果报告中与已证明淘汰分开统计。

### 2026-07-18：内层采用证明分数与修复风险双分数

内层 Gurobi 的正式优化目标为 `F_relax`：基准复合目标在无站台等待松弛模型
上的可证明下界。`Cmax`、active-slot、route-travel 等保留下来的目标项沿用基准
case 的实际系数；依赖已删除约束的正罚项，只有在能证明其松弛表达仍是原项
下界时才保留，否则以经过证明的下界（通常为 0）替代。`F_relax` 的表达式和
证明标签写入模型 manifest，只有该目标产生的 `ObjBound` 可参与硬剪枝。

同时计算独立的 `repair_risk`，用于衡量候选回到完整外层后的可修复性，包括
站台到达重叠、潜在等待拥堵、容量/时间余量、受保护 warm 结构扰动和邻域距离。
`repair_risk` 不写入认证目标，不参与 `ObjBound`、永久剪枝、正式 incumbent
接受或 time-to-target 判定；它只用于同一次内层求解所得候选之间的调度和
排序。日志必须分别输出 `F_relax`、其 bound、各风险分量和最终排序键，禁止
把二者合并成看似可证明的单一分数。

### 2026-07-18：单次内层求解使用自然精英池

每次 procedure 的 inner MIP 通过 `MIPSOL` callback 收集求解过程中自然出现的
候选，不开启额外的 Gurobi solution-pool 搜索，也不为填满池子延长 inner
time limit。候选按完整 outer-fixed 结构投影计算确定性 hash；重复 hash 只保留
更优记录，精英池最多保留 8 个互异结构。

池内维护 `(F_relax_value, repair_risk)` 的非支配候选，并在超过容量时按确定性
多样性规则裁剪。`F_relax_value` 是松弛可行解值，不等同于证明下界；不得因为
该值不优于 incumbent 就硬淘汰。只有候选专属的认证 LB 或适用的 Gurobi
`ObjBound` 达到 incumbent cutoff 时，才允许证明性删除。

inner solve 结束后，从仍有改进可能的非支配候选中，按 `repair_risk`、
`F_relax_value` 和结构 hash 的确定性顺序选出一个送入完整外层。每个 procedure
仍最多触发一次首次 outer 认证；其后的保留预算重试属于同一候选，不计作新的
候选提交。

### 2026-07-18：canonical warm 禁止计时前优化或修复

Gurobi 基准路径、TRA-Gurobi 和后续 TRA-Fast 读取同一份 canonical warm
artifact。计时前只允许确定性生成、序列化、hash 校验和不改变变量值的可行性
检查；不得调用 Gurobi 求解、repair、局部搜索或任何会改进 warm 的启发式。

warm 使用的 stack、station-task、inventory action 和 route arc 必须并入共享
manifest 的 protected 集合，使它们不被 top-k、KNN、dominance 或 route prune
删除。`protected` 只表示域保留，不表示在求解中固定；轮转和完整外层都可以
选择离开 warm 结构。

若 canonical warm 已经是完整原模型可行解，则它作为 `t=0` incumbent；若它
只是部分 MIP start 或未通过完整 verifier，则正式搜索从“无可行 incumbent”
启动。此时第一轮不施加 incumbent improvement cutoff，首个完整外层可行解建立
incumbent；投影 MIP start、修复与认证的全部时间均从第一轮开始计时。

### 2026-07-18：九类结构采用最小非冗余投影固定

论文九个 outer-fixed 变量在当前模型中按语义投影，而不是按名称硬凑九个新变量：

| 论文变量 | 当前 outer-fixed 投影 |
| --- | --- |
| `kappa` pod used | 任一 `S_visit[(slot,stack),station]` 激活所派生的 stack-used 状态 |
| `alpha` robot-pod | `R_assign`，M1-M9 由 `slot_robot` 承载 |
| `vartheta` pod-station | `S_visit`，链接到 `pair_activate` |
| `beta` cluster-station | `Y_station`，即 `sum_rank y[slot,station,rank]` |
| `psi` order-cluster | `X_group` 的 `x[work_unit,slot]` |
| `xi` cluster-pod | slot-stack active 状态，由 `S_visit/Z_inventory` 一致派生 |
| `theta` picked quantity | `Z_inventory` 的 flip/sort/carry/hit/noise/flip-hit 选择及数量表达式 |
| `mu` SKU picked flag | 由 `hit/flip_hit` 和 SKU cover 关系派生的命中投影 |
| `gamma` station workload | 由固定 `X_group/Z_inventory/S_visit` 唯一确定的处理工作量表达式 |

outer 固定最小非冗余载体 `x + S_visit + Y_station + Z_inventory + R_assign`；
`a/sku_use/pair_activate/passX/route_owner/kappa/xi/mu/gamma` 由原约束推导并做
一致性断言，不再重复施加可能相互矛盾的固定。outer 明确释放：

- `Y_rank`、station arrival/finish clocks 和所有站台先后/等待变量；
- `route_arc`、`route_time`、`route_load`、`route_finish`；
- `arrival/start/finish`、order time-window 连续变量和 `Cmax`。

这对应论文释放 `delta/phi/rho/sigma`。旧 evaluator 的
`fixed_station_rank_by_order_slot`、`fixed_route_*sequence*` 和 route replay 不得
进入正式 outer 路径。

### 2026-07-18：内层是完整模型的显式无等待松弛

inner model 直接消费共享 manifest 和 full template 的 prepared domain，不运行
自己的候选生成或剪枝。它保留 X、inventory、station assignment、visit、robot
assignment、route arc、pickup-delivery、route time/load 和所有仍然有效的
结构约束。站台部分改为：

1. 每个 active slot 选择一个 `Y_station`，但不创建 `Y_rank`；
2. `arrival^R` 仍由所有 active delivery 的 route time 下界约束；
3. 强制 `start^R = arrival^R`，并以原处理工作量定义 `finish^R`；
4. 删除 station rank unique/no-hole、station clocks、前序完成后开始、等待时间和
   依赖这些变量的对称约束；
5. 保留不依赖 station waiting 的 workload、intrinsic arrival、capacity、route
   和 order-arrival 下界。

必须通过双向差分测试证明：任意 full-feasible 解投影后都是 inner-feasible；对
同一固定结构，`F_relax_opt <= F_outer_opt + tolerance`。未通过该测试的约束不能
进入 inner 的证明性模型，只能降为 `repair_risk`。

### 2026-07-18：使用预编译持久模板而非逐轮重建

计时前分别构造一个 full outer template 和一个 no-wait inner template；只允许
`model.update()` 和静态 fingerprint，不允许 `optimize()`、`presolve()` 或预求
根松弛。正式轮转在持久模板上通过变量 LB/UB 和一条可替换的 local-branching
约束更新固定投影，避免当前 `solve_compiled()` 每次复制整模和重新添加大量固定
约束的开销。

每次求解前必须恢复所有临时变量界、Start、cutoff、time limit、callback state
和 local-branching 行，并校验模板 fingerprint 未漂移。模型更新、reset、MIP
start 投影和求解全部计入正式 runtime。禁止跨候选保留会改变数学模型的局部
cut；仅 Gurobi 在一次 `optimize()` 内部生成的搜索状态可由其自身管理。

### 2026-07-18：轮转控制保持严格 F1-F2-F3 周期

轮转控制器按 `F1(S_visit) -> F2(X_group) -> F3(R_assign)` 周期选择本轮释放
变量；每个 inner Gurobi 固定另外两个核心载体。预算 scheduler 只能调整各过程
time slice、N1/N2/N3 层级和延迟候选重试时机，不能长期跳过某个核心块或把
`GLOBAL/XYZ` 当作第四个 procedure。incumbent 一旦改进，三个过程的邻域层级
都重置为 N1，所有依赖旧 incumbent 的 deferred/LB 证书失效并重新计算。

### 2026-07-18：正式 runner 与目标文件物理隔离

正式 TRA 进程只读取不含 Cmax 的两类输入：sanitized domain policy 和仅含旧
runtime 的 budget policy。历史 `gurobi_summary.json`、solution export、当前
`tra_gurobi.py` 中的 `TARGET_CMAX` 常量、target probe、known-target guidance、
`BestObjStop` 及 controlled-release target polish 均不得被正式 runner 导入。

启动时执行 fail-closed 审计：任何 finite target、`BestObjStop`、历史结构路径、
route replay 或 solution export 字段都会使运行在计时前失败。正式运行结束并
冻结日志后，独立验收进程才可读取 4.2 Cmax。

### 2026-07-18：用可回放可行解事件计算首次命中时间

inner 解不产生 time-to-target 事件。每次 full outer 的 `MIPSOL` callback 对每个
新复合 incumbent 记录正式起点后的 wall timestamp、复合目标、solver Cmax、
结构 hash 和稀疏解快照。outer 返回后、进入下一 procedure 前，在正式计时内按
时间顺序运行内部只读 verifier；通过者追加到 append-only
`feasible_solution_events.jsonl`，失败者保留错误证据但不参与命中计算。

事件记录和验证不知道 4.2 Cmax，也不得只保留“当前猜测目标附近”的解。运行后
验收器在已验证事件中寻找首次 `verified_cmax == old_4.2_cmax`，采用该事件原始
callback timestamp 作为 time-to-target。最终 incumbent 和所有事件都必须在
`B_case` 内产生；计时结束后的补解或补验证不能生成合格事件。
