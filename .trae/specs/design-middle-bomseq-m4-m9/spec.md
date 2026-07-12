# Middle M4-M9 BOMSeq 算例设计 Spec

## Why
当前 middle 系列只有 M1-M3 accepted，M4 仍未达标，M5-M9 尚未逐个验收。需要继续按固定地图序列和 BOM/SKU 规模序列设计 M4-M9，并把每个通过的 case 写入可复现 JSON 和汇总表。

## What Changes
- 继续设计并验收 middle BOM-sequence 系列 M4-M9。
- 地图按 `4x4 -> 5x4 -> 4x6 -> 5x6 -> 4x8 -> 5x8`，每个地图约 2 个 case 的节奏组织。
- BOM 数参考 stacks 系列从 6 起步，当前序列为 `6,6,6,6,8,8,8,8,10`。
- 每 BOM SKU 数按 `10,14,18,22,7,10,14,18,22`。
- 每个通过的 case 必须写入 `tmp/middle_stack_bomseq_runtime_configs.json`，并同步完整结果字段。
- 尽量避免使用 KNN pickup 剪枝；允许安全 route 剪枝、容量剪枝、time-window 剪枝、warm-start 注入和有效不等式。

## Impact
- Affected specs: middle M1-M9 benchmark design and acceptance.
- Affected code: `experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json`, `tmp/middle_stack_bomseq_runtime_configs.json`, `experiments/run_global_xyzu.py`, `Gurobi/global_xyzu.py`, `result/middle_bomseq_*` outputs.

## ADDED Requirements
### Requirement: Middle Case Progression
The system SHALL continue M4-M9 case design using the current middle stack-grid map layout.

#### Scenario: Case sequence follows requested maps
- **WHEN** M4-M9 are configured
- **THEN** their maps SHALL follow the planned `5x4, 4x6, 5x6, 4x8, 5x8` progression while preserving already accepted M1-M3 records.

### Requirement: Accepted Case Result Ledger
The system SHALL write every accepted case result into `tmp/middle_stack_bomseq_runtime_configs.json`.

#### Scenario: A case passes acceptance
- **WHEN** a middle case satisfies gap, runtime, audit, and TRA verification constraints
- **THEN** the JSON SHALL include its runnable config, command, output root, and table fields: case, BOM数, 每BOM SKU数, 总SKU数, tote, stack, robot, station, 总需求量, 变量数, 约束数, 命中stack数, subtask数, flip的tote数, sort的tote数, 上界, 下界, cmax, gap, runtime, status.

### Requirement: Avoid KNN Pruning
The system SHALL avoid pickup KNN pruning for accepted rows unless explicitly marked as an exploratory rejected probe.

#### Scenario: Accepted row command is recorded
- **WHEN** command metadata is written for an accepted case
- **THEN** `route_pickup_neighbor_limit` SHALL be `0` or absent with documented equivalence to disabled KNN.

### Requirement: M4 First
The system SHALL resolve M4 before advancing to M5-M9.

#### Scenario: M4 is not accepted
- **WHEN** M4 does not meet the acceptance criteria
- **THEN** implementation SHALL continue M4 tuning and record rejected probes before running later case acceptance.

## MODIFIED Requirements
### Requirement: Middle Acceptance Criteria
The accepted middle cases SHALL have Gurobi incumbent, `model_gap <= 1%`, clean audit, TRA verification PASS, and runtime within the intended case window where applicable.

## REMOVED Requirements
### Requirement: Use KNN to reduce model size
**Reason**: The user requested to avoid KNN pruning to reduce risk that TRA later finds edges excluded by Gurobi.
**Migration**: Prefer safe route arc pruning, route time-window pruning, route workload lower bounds, capacity lower bounds, and warm-start improvements.
