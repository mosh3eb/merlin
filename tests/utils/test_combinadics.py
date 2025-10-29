import math

import perceval as pcvl
import pytest

from merlin import QuantumLayer
from merlin.utils.combinadics import Combinadics


def _enumerate_states(combo: Combinadics, total: int):
    states = [combo.index_to_fock(i) for i in range(total)]
    assert len(states) == total
    # ensure descending lexicographic order
    for prev, curr in zip(states, states[1:], strict=False):
        assert prev >= curr
    # check bidirectional consistency
    for idx, state in enumerate(states):
        assert combo.fock_to_index(state) == idx
    return states


def test_dual_rail_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("dual_rail", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == 2**n
    states = _enumerate_states(combo, expected_total)
    assert len(states) == expected_total
    # each pair must contain exactly one photon
    for state in states:
        for i in range(0, m, 2):
            a, b = state[i], state[i + 1]
            assert a in (0, 1) and b in (0, 1) and a + b == 1


def test_unbunched_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("unbunched", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == math.comb(m, n)
    states = _enumerate_states(combo, expected_total)
    assert len(states) == expected_total
    # unbunched: collision free
    for state in states:
        assert sum(state) == n
        for occ in state:
            assert occ in (0, 1)


def test_fock_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("fock", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == math.comb(n + m - 1, m - 1)
    states = _enumerate_states(combo, expected_total)
    assert len(states) == expected_total
    # fock states: non-negative integers summing to n
    for state in states:
        assert sum(state) == n
        for occ in state:
            assert occ >= 0


def test_dual_rail_requires_twice_as_many_modes():
    with pytest.raises(ValueError):
        Combinadics("dual_rail", n=4, m=6)


@pytest.mark.parametrize(
    "scheme,no_bunching,n,m",
    [
        ("unbunched", True, 4, 8),
        ("fock", False, 4, 8),
        ("unbunched", True, 5, 3),
        ("fock", False, 5, 3),
        ("unbunched", True, 2, 10),
        ("fock", False, 2, 10),
    ],
)
def test_iteration_order_matches_quantum_layer(
    scheme: str, no_bunching: bool, n: int, m: int
):
    n, m = 4, 8
    ql = QuantumLayer(
        input_size=0,
        circuit=pcvl.Circuit(m),
        n_photons=n,
        no_bunching=no_bunching,
    )

    mapped_keys = [
        tuple(state) for state in ql.computation_process.simulation_graph.mapped_keys
    ]

    combo = Combinadics(scheme, n=n, m=m)
    states = [combo.index_to_fock(i) for i in range(combo.compute_space_size())]

    assert mapped_keys == states
