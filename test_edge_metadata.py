from types import SimpleNamespace

import pytest

from grid_editor5 import (
    assign_element_edge_metadata,
    ordered_edge_chain,
    ordered_capped_boundary_gap,
    ordered_extended_boundary_topology,
    patch_corner_count_error,
    ordered_three_edge_chain,
    two_edge_corner_nodes,
    validate_boundary_chain,
    validate_single_element_patch,
)


def make_edge(start, end):
    return SimpleNamespace(
        vertices=[start, end],
        local_nodes_index=None,
        element_index=None,
        element_side=None,
    )


def make_selected_edge(start, end, element_index, uv_index):
    return SimpleNamespace(
        nodes=[SimpleNamespace(index=start), SimpleNamespace(index=end)],
        element_index=element_index,
        uv_index=uv_index,
    )


def test_rejects_two_adjacent_edges_from_the_same_element():
    edges = [make_selected_edge(1, 2, 7, 1), make_selected_edge(2, 3, 7, 2)]

    assert "different elements" in validate_single_element_patch(edges)


def test_accepts_two_adjacent_edges_from_different_elements():
    edges = [make_selected_edge(1, 2, 7, 1), make_selected_edge(2, 3, 8, 2)]

    assert validate_single_element_patch(edges) is None


def test_rejects_same_element_edges_without_exactly_one_shared_node():
    disjoint = [make_selected_edge(1, 2, 7, 1), make_selected_edge(3, 4, 7, 2)]
    duplicate = [make_selected_edge(1, 2, 7, 1), make_selected_edge(2, 1, 7, 2)]

    assert "exactly one node" in validate_single_element_patch(disjoint)
    assert "exactly one node" in validate_single_element_patch(duplicate)


def test_rejects_two_edges_with_the_same_local_direction():
    edges = [make_selected_edge(1, 2, 7, 1), make_selected_edge(2, 3, 8, 1)]

    assert "different local directions" in validate_single_element_patch(edges)


def test_extracts_shared_and_outer_nodes_in_edge_order():
    edge0 = make_selected_edge(10, 11, 7, 1)
    edge1 = make_selected_edge(12, 10, 8, 2)

    shared, outer0, outer1 = two_edge_corner_nodes(edge0, edge1)

    assert shared.index == 10
    assert outer0.index == 11
    assert outer1.index == 12


def test_orders_valid_three_edge_chain_independently_of_selection_order():
    edges = [
        make_selected_edge(2, 3, 8, 2),
        make_selected_edge(3, 4, 9, 1),
        make_selected_edge(1, 2, 7, 1),
    ]

    ordered_nodes, ordered_edges = ordered_three_edge_chain(edges)

    assert [node.index for node in ordered_nodes] == [1, 2, 3, 4]
    assert [edge.uv_index for edge in ordered_edges] == [1, 2, 1]
    assert validate_single_element_patch(edges) is None
    assert patch_corner_count_error(edges, []) is None


@pytest.mark.parametrize("edge_count", [1, 2, 5, 8])
def test_orders_arbitrary_boundary_chain_lengths(edge_count):
    nodes = [SimpleNamespace(index=index) for index in range(edge_count + 1)]
    chain_edges = [
        SimpleNamespace(nodes=[nodes[index], nodes[index + 1]])
        for index in range(edge_count)
    ]
    supplied_edges = list(reversed(chain_edges))
    for index in range(0, len(supplied_edges), 2):
        supplied_edges[index].nodes.reverse()

    ordered_nodes, ordered_edges = ordered_edge_chain(supplied_edges)

    assert ordered_nodes == nodes
    assert ordered_edges == chain_edges


def test_same_direction_edges_are_valid_only_for_boundary_chain_operation():
    edges = [
        make_selected_edge(178, 179, 7, 1),
        make_selected_edge(179, 180, 8, 1),
    ]

    assert "different local directions" in validate_single_element_patch(edges)
    assert validate_boundary_chain(edges) is None


def test_orders_capped_boundary_gap_independently_of_selection_order():
    nodes = {index: SimpleNamespace(index=index) for index in range(5)}

    def edge(start, end, uv_index):
        return SimpleNamespace(
            nodes=[nodes[start], nodes[end]], uv_index=uv_index
        )

    main0 = edge(1, 2, 1)
    main1 = edge(3, 2, 1)
    start_side = edge(1, 0, 2)
    end_side = edge(4, 3, 2)

    gap = ordered_capped_boundary_gap(
        [end_side, main1, start_side, main0]
    )

    assert [node.index for node in gap.inner_nodes] == [1, 2, 3]
    assert gap.inner_edges == [main0, main1]
    assert gap.outer_start is nodes[0]
    assert gap.outer_end is nodes[4]
    assert gap.start_side_edge is start_side
    assert gap.end_side_edge is end_side


def test_explicit_main_uv_resolves_symmetric_multi_edge_one_cap():
    nodes = {index: SimpleNamespace(index=index) for index in range(5)}

    def edge(start, end, uv_index):
        return SimpleNamespace(
            nodes=[nodes[start], nodes[end]], uv_index=uv_index
        )

    uv1_edges = [edge(0, 1, 1), edge(1, 2, 1)]
    uv2_edges = [edge(0, 3, 2), edge(3, 4, 2)]
    selected = [uv2_edges[1], uv1_edges[1], uv2_edges[0], uv1_edges[0]]

    with pytest.raises(ValueError, match="Ambiguous extended patch"):
        ordered_extended_boundary_topology(selected)

    topology = ordered_extended_boundary_topology(
        selected, main_uv_index=1
    )
    assert topology.inner_edges == uv1_edges
    assert topology.start_cap_edges == uv2_edges
    assert [node.index for node in topology.start_cap_nodes] == [0, 3, 4]


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_orders_one_cap_topology_at_oriented_chain_end(cap_at_start):
    nodes = {index: SimpleNamespace(index=index) for index in range(5)}

    def edge(start, end, uv_index):
        return SimpleNamespace(
            nodes=[nodes[start], nodes[end]], uv_index=uv_index
        )

    main0 = edge(2, 1, 1)
    main1 = edge(3, 2, 1)
    cap = edge(0, 1, 2) if cap_at_start else edge(4, 3, 2)

    topology = ordered_extended_boundary_topology(
        [main1, cap, main0], main_uv_index=1
    )

    assert [node.index for node in topology.inner_nodes] == [1, 2, 3]
    assert topology.inner_edges == [main0, main1]
    if cap_at_start:
        assert topology.start_cap_edge is cap
        assert topology.outer_start_node is nodes[0]
        assert topology.end_cap_edge is None
        assert topology.outer_end_node is None
    else:
        assert topology.start_cap_edge is None
        assert topology.outer_start_node is None
        assert topology.end_cap_edge is cap
        assert topology.outer_end_node is nodes[4]


@pytest.mark.parametrize("cap_uv_index", [1, 2])
def test_rejects_cap_attached_to_interior_chain_node(cap_uv_index):
    nodes = {index: SimpleNamespace(index=index) for index in range(5)}
    edges = [
        SimpleNamespace(nodes=[nodes[1], nodes[2]], uv_index=1),
        SimpleNamespace(nodes=[nodes[2], nodes[3]], uv_index=1),
        SimpleNamespace(nodes=[nodes[2], nodes[4]], uv_index=cap_uv_index),
    ]

    with pytest.raises(ValueError):
        ordered_extended_boundary_topology(edges)


@pytest.mark.parametrize(
    "edge_specs",
    [
        [(0, 1, 1), (2, 3, 1), (4, 0, 2), (3, 5, 2)],
        [(0, 1, 1), (1, 2, 1), (1, 3, 1), (4, 0, 2), (2, 5, 2)],
        [(0, 1, 1), (1, 2, 1), (3, 0, 2), (2, 3, 2)],
    ],
)
def test_rejects_invalid_capped_boundary_gap(edge_specs):
    nodes = {
        index: SimpleNamespace(index=index)
        for index in {value for start, end, uv in edge_specs for value in (start, end)}
    }
    edges = [
        SimpleNamespace(nodes=[nodes[start], nodes[end]], uv_index=uv)
        for start, end, uv in edge_specs
    ]

    with pytest.raises(ValueError):
        ordered_capped_boundary_gap(edges)


def test_ordered_chain_preserves_original_node_and_edge_objects():
    nodes = [SimpleNamespace(index=index) for index in range(4)]
    edges = [
        SimpleNamespace(nodes=[nodes[2], nodes[3]]),
        SimpleNamespace(nodes=[nodes[2], nodes[1]]),
        SimpleNamespace(nodes=[nodes[0], nodes[1]]),
    ]

    ordered_nodes, ordered_edges = ordered_edge_chain(edges)

    assert all(actual is expected for actual, expected in zip(ordered_nodes, nodes))
    assert ordered_edges[0] is edges[2]
    assert ordered_edges[1] is edges[1]
    assert ordered_edges[2] is edges[0]


@pytest.mark.parametrize(
    "pairs",
    [
        [(0, 1), (2, 3)],
        [(0, 1), (1, 2), (2, 0)],
        [(0, 1), (1, 2), (1, 3)],
        [(0, 1), (1, 0)],
    ],
)
def test_rejects_non_open_or_duplicate_boundary_chains(pairs):
    nodes = {
        index: SimpleNamespace(index=index)
        for index in {vertex for pair in pairs for vertex in pair}
    }
    edges = [
        SimpleNamespace(nodes=[nodes[start], nodes[end]])
        for start, end in pairs
    ]

    with pytest.raises(ValueError):
        ordered_edge_chain(edges)


def test_rejects_three_edge_chain_without_alternating_directions():
    edges = [
        make_selected_edge(1, 2, 7, 1),
        make_selected_edge(2, 3, 8, 1),
        make_selected_edge(3, 4, 9, 2),
    ]

    assert "alternate uv_index" in validate_single_element_patch(edges)


def test_rejects_three_edges_without_four_distinct_vertices():
    edges = [
        make_selected_edge(1, 2, 7, 1),
        make_selected_edge(2, 3, 8, 2),
        make_selected_edge(3, 1, 9, 1),
    ]

    assert "exactly four distinct vertices" in validate_single_element_patch(edges)


def test_one_edge_requires_exactly_two_new_points():
    edge = make_selected_edge(1, 2, 7, 1)

    assert "exactly two new points" in patch_corner_count_error([edge], [object()])
    assert patch_corner_count_error([edge], [object(), object()]) is None


def test_two_edges_require_exactly_one_new_point():
    edges = [make_selected_edge(1, 2, 7, 1), make_selected_edge(2, 3, 8, 2)]

    assert "exactly one new point" in patch_corner_count_error(edges, [])
    assert "exactly one new point" in patch_corner_count_error(
        edges, [object(), object()]
    )
    assert patch_corner_count_error(edges, [object()]) is None


@pytest.mark.parametrize(
    "edges, order, vertices",
    [
        (
            [make_edge(10, 11), make_edge(11, 12), make_edge(12, 13), make_edge(13, 10)],
            [0, 1, 2, 3],
            [10, 11, 12, 13],
        ),
        (
            [make_edge(21, 22), make_edge(22, 23), make_edge(23, 20), make_edge(20, 21)],
            [3, 0, 1, 2],
            [20, 21, 22, 23],
        ),
    ],
)
def test_assigns_zero_based_metadata_after_final_edge_order(edges, order, vertices):
    new_element_index = 17

    assign_element_edge_metadata(edges, order, new_element_index, vertices)

    for side, edge_index in enumerate(order):
        edge = edges[edge_index]
        expected_local_nodes = [side, (side + 1) % 4]
        assert edge.element_side == side
        assert edge.local_nodes_index == expected_local_nodes
        assert edge.vertices == [
            vertices[expected_local_nodes[0]],
            vertices[expected_local_nodes[1]],
        ]
        assert edge.element_index == new_element_index
