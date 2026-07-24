from __future__ import annotations

from types import SimpleNamespace

from Gurobi.tra_outer_search import configure_outer_certification_search


def test_outer_certification_search_is_exact_and_incumbent_focused() -> None:
    params = SimpleNamespace(
        MIPFocus=0,
        MIPGap=0.1,
        PoolSearchMode=2,
        Heuristics=0.05,
        PumpPasses=0,
        StartNodeLimit=500,
        RINS=-1,
    )

    configure_outer_certification_search(SimpleNamespace(Params=params))

    assert params.MIPFocus == 1
    assert params.MIPGap == 0.0
    assert params.PoolSearchMode == 0
    assert params.Heuristics == 0.35
    assert params.PumpPasses == 20
    assert params.StartNodeLimit == 2000
    assert params.RINS == 10
