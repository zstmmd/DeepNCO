from __future__ import annotations

from types import SimpleNamespace

from Gurobi.tra_candidate_archive import CandidateArchive, fixed_blocks_compatible
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_scheduler import ProcedureStep


def _shell(name: str, x_group: dict[str, int]):
    return SimpleNamespace(
        sha256=name,
        projection=SimpleNamespace(
            x_group=x_group,
            s_visit={(0, 1): 0},
            r_assign={0: 0},
        ),
    )


def _candidate(name: str, x_group: dict[str, int], score: float):
    return SimpleNamespace(
        shell=_shell(name, x_group),
        relaxed_objective=10.0,
        repair_risk=SimpleNamespace(total=1.0),
        comproc=SimpleNamespace(
            feasible=True,
            recourse_score=score,
            projected_cmax=score,
            projected_objective=score,
            verified_cmax=score,
        ),
    )


def test_archive_prefers_released_block_diversity_before_comproc_score() -> None:
    archive = CandidateArchive()
    reference = _shell("reference", {"a": 0, "b": 0, "c": 1, "d": 1})
    near = _candidate("near", {"a": 1, "b": 0, "c": 0, "d": 1}, 11.0)
    far = _candidate("far", {"a": 1, "b": 1, "c": 0, "d": 0}, 20.0)
    step = ProcedureStep(2, 1, Procedure.F2, NeighborhoodLevel.N2)

    archive.remember(reference, step, (near, far))

    ranked = archive.ranked(Procedure.F2, reference)
    assert [item.candidate.shell.sha256 for item in ranked] == ["far", "near"]


def test_archive_excludes_the_submitted_shell_and_prunes_explored_shells() -> None:
    archive = CandidateArchive()
    reference = _shell("reference", {"a": 0, "b": 1})
    submitted = _candidate("submitted", {"a": 1, "b": 0}, 10.0)
    runner_up = _candidate("runner-up", {"a": 1, "b": 0}, 11.0)
    step = ProcedureStep(2, 1, Procedure.F2, NeighborhoodLevel.N2)

    archive.remember(
        reference,
        step,
        (submitted, runner_up),
        excluded_hashes=("submitted",),
    )

    assert archive.count == 1
    assert archive.ranked(
        Procedure.F2,
        reference,
        excluded_hashes=("runner-up",),
    ) == ()
    assert archive.empty


def test_archive_requires_the_two_fixed_blocks_to_match_the_current_shell() -> None:
    current = _shell("current", {"a": 0, "b": 1})
    changed_x = _shell("changed-x", {"a": 1, "b": 0})
    changed_r = _shell("changed-r", {"a": 0, "b": 1})
    changed_r.projection.r_assign = {0: 1}

    assert fixed_blocks_compatible(Procedure.F2, current, changed_x)
    assert not fixed_blocks_compatible(Procedure.F2, current, changed_r)
    assert not fixed_blocks_compatible(Procedure.F3, current, changed_x)

    archive = CandidateArchive()
    f3_candidate = _candidate("old-f3", {"a": 1, "b": 0}, 10.0)
    archive.remember(
        changed_x,
        ProcedureStep(3, 1, Procedure.F3, NeighborhoodLevel.N3),
        (f3_candidate,),
    )
    assert archive.ranked(Procedure.F3, current) == ()


def test_archive_can_require_candidates_from_the_current_parent_shell() -> None:
    archive = CandidateArchive()
    old_parent = _shell("old-parent", {"a": 0, "b": 1})
    current_parent = _shell("current-parent", {"a": 0, "b": 1})
    old_candidate = _candidate("old-candidate", {"a": 1, "b": 0}, 10.0)
    current_candidate = _candidate("current-candidate", {"a": 1, "b": 0}, 11.0)
    step = ProcedureStep(2, 1, Procedure.F2, NeighborhoodLevel.N2)

    archive.remember(old_parent, step, (old_candidate,))
    archive.remember(current_parent, step, (current_candidate,))

    ranked = archive.ranked(
        Procedure.F2,
        current_parent,
        required_reference_sha256=current_parent.sha256,
    )
    assert [item.candidate.shell.sha256 for item in ranked] == [
        "current-candidate"
    ]
