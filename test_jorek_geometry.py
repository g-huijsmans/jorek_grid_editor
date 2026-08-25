import numpy as np

from jorek_geometry import (
    basis_vector_from_scene,
    pair_corners_to_edge_endpoints,
    quad_area,
    reorder_corners,
    triangle_area,
)


def test_pairs_corners_with_nearest_edge_endpoints_regardless_of_input_order():
    node0 = np.array([0.0, 0.0])
    node1 = np.array([10.0, 0.0])
    near_node1 = np.array([9.0, 5.0])
    near_node0 = np.array([1.0, 5.0])

    new_at_node0, new_at_node1 = pair_corners_to_edge_endpoints(
        node0, node1, near_node1, near_node0
    )

    np.testing.assert_array_equal(new_at_node0, near_node0)
    np.testing.assert_array_equal(new_at_node1, near_node1)


def test_basis_vector_from_scene_preserves_scaling_and_origin():
    node_position = np.array([120.0, -30.0])
    handle_position = np.array([150.0, 50.0])

    np.testing.assert_array_equal(
        basis_vector_from_scene(node_position, handle_position, 100.0),
        np.array([0.3, 0.8]),
    )


def test_triangle_area_preserves_orientation_sign():
    origin = np.array([0.0, 0.0])
    right = np.array([2.0, 0.0])
    top = np.array([0.0, 3.0])

    assert triangle_area(origin, right, top) == 3.0
    assert triangle_area(origin, top, right) == -3.0


def test_quad_area_preserves_orientation_sign():
    corners = [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([2.0, 3.0]),
        np.array([0.0, 3.0]),
    ]

    assert quad_area(*corners) == 6.0
    assert quad_area(*reversed(corners)) == -6.0


def test_reorder_corners_preserves_both_existing_results():
    x1 = np.array([0.0, 0.0])
    x2 = np.array([2.0, 0.0])
    x3 = np.array([2.0, 2.0])

    assert reorder_corners(x1, x2, x3, np.array([0.0, 2.0])) == [0, 1, 3, 4]
    assert reorder_corners(x1, x2, x3, np.array([3.0, 1.0])) == [0, 1, 4, 3]
