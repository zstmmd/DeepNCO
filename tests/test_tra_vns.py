from __future__ import annotations

from dataclasses import dataclass

import pytest

from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure, validate_transition
from Gurobi.tra_projection import CoreProjection, ProjectionRegistry, StructuralShell
from Gurobi.tra_vns import PaperVNSGenerator, rotating_search_seed


@dataclass(frozen=True)
class _Var:
    VarName: str


def _generator_and_shell():
    units = ("u1", "u2", "u3", "u4")
    slots = (10, 11, 12, 13)
    stacks = (20, 21, 22, 23)
    stations = (30, 31)
    robots = (40, 41, 42)
    families = {
        "x": {
            (unit, slot): _Var(f"x[{unit},{slot}]")
            for unit in units
            for slot in (10, 11)
        },
        "pair_activate": {
            (slot, stack, station): _Var(f"pair[{slot},{stack},{station}]")
            for slot, stack in zip(slots, stacks)
            for station in stations
        },
        "slot_robot": {
            (slot, robot): _Var(f"robot[{slot},{robot}]")
            for slot in slots
            for robot in robots
        },
    }
    shell = StructuralShell(
        projection=CoreProjection(
            x_group={"u1": 10, "u2": 11, "u3": 10, "u4": 11},
            s_visit={
                (10, 20): 30,
                (11, 21): 31,
                (12, 22): 30,
                (13, 23): 31,
            },
            r_assign={10: 40, 11: 41, 12: 40, 13: 41},
        )
    )
    return PaperVNSGenerator(ProjectionRegistry(families)), shell


@pytest.mark.parametrize("procedure", list(Procedure))
@pytest.mark.parametrize("neighborhood", list(NeighborhoodLevel))
def test_vns_seeds_follow_the_exact_released_block_radius(procedure, neighborhood) -> None:
    generator, shell = _generator_and_shell()

    seeds = generator.generate(
        shell,
        procedure=procedure,
        neighborhood=neighborhood,
        limit=2,
    )

    assert seeds
    for seed in seeds:
        audit = validate_transition(
            shell.projection,
            seed.projection,
            procedure,
            neighborhood,
        )
        assert audit.raw_one_hot_hamming <= neighborhood.raw_hamming_limit
        assert seed.values_by_name


def test_vns_offset_rotates_to_a_disjoint_deterministic_seed_window() -> None:
    generator, shell = _generator_and_shell()

    first = generator.generate(
        shell,
        procedure=Procedure.F3,
        neighborhood=NeighborhoodLevel.N1,
        limit=2,
        offset=0,
    )
    second = generator.generate(
        shell,
        procedure=Procedure.F3,
        neighborhood=NeighborhoodLevel.N1,
        limit=2,
        offset=2,
    )

    assert len(first) == len(second) == 2
    assert {seed.sha256 for seed in first}.isdisjoint(
        seed.sha256 for seed in second
    )
    assert second == generator.generate(
        shell,
        procedure=Procedure.F3,
        neighborhood=NeighborhoodLevel.N1,
        limit=2,
        offset=2,
    )


def test_f3_n1_seeds_expand_underused_robot_support_first() -> None:
    generator, shell = _generator_and_shell()
    crowded = StructuralShell(
        projection=shell.projection.replace_block(
            "r_assign",
            {10: 40, 11: 40, 12: 40, 13: 41},
        )
    )

    seed = generator.generate(
        crowded,
        procedure=Procedure.F3,
        neighborhood=NeighborhoodLevel.N1,
        limit=1,
    )[0]

    assert list(seed.projection.r_assign.values()).count(42) == 1
    assert list(seed.projection.r_assign.values()).count(40) == 2


def test_f3_n3_seed_reduces_robot_count_imbalance() -> None:
    generator, shell = _generator_and_shell()
    crowded = StructuralShell(
        projection=shell.projection.replace_block(
            "r_assign",
            {10: 40, 11: 40, 12: 40, 13: 41},
        )
    )

    seed = generator.generate(
        crowded,
        procedure=Procedure.F3,
        neighborhood=NeighborhoodLevel.N3,
        limit=1,
        balance_support=True,
    )[0]
    counts = [
        list(seed.projection.r_assign.values()).count(robot_id)
        for robot_id in (40, 41, 42)
    ]

    assert max(counts) - min(counts) < 3


def test_f3_n3_preserves_label_counts_before_full_incumbent_exists() -> None:
    generator, shell = _generator_and_shell()

    seed = generator.generate(
        shell,
        procedure=Procedure.F3,
        neighborhood=NeighborhoodLevel.N3,
        limit=1,
        balance_support=False,
    )[0]

    assert sorted(seed.projection.r_assign.values()) == sorted(
        shell.projection.r_assign.values()
    )


def test_rotating_search_seed_preserves_first_window_and_changes_later_windows() -> None:
    assert rotating_search_seed(42, offset=0, width=4) == 42
    assert rotating_search_seed(42, offset=4, width=4) != 42
    assert rotating_search_seed(42, offset=4, width=4) == rotating_search_seed(
        42,
        offset=4,
        width=4,
    )
