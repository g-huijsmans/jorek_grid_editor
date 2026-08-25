import numpy as np


def basis_vector_from_scene(node_position, handle_position, scaling):
    """Convert a basis-vector endpoint from scene units to JOREK grid units."""
    return (np.asarray(handle_position) - np.asarray(node_position)) / scaling


def pair_corners_to_edge_endpoints(node0, node1, corner0, corner1):
    """Pair two new corners with edge endpoints by minimum total distance."""
    node0 = np.asarray(node0)
    node1 = np.asarray(node1)
    corner0 = np.asarray(corner0)
    corner1 = np.asarray(corner1)

    cost1 = np.linalg.norm(node0 - corner0) + np.linalg.norm(node1 - corner1)
    cost2 = np.linalg.norm(node0 - corner1) + np.linalg.norm(node1 - corner0)
    if cost1 <= cost2:
        return corner0, corner1
    return corner1, corner0


def triangle_area(x1, x2, x3):
    return 0.5 * (
        x1[0] * (x2[1] - x3[1])
        + x2[0] * (x3[1] - x1[1])
        + x3[0] * (x1[1] - x2[1])
    )


def quad_area(x1, x2, x3, x4):
    return 0.5 * (
        x1[0] * x2[1]
        + x2[0] * x3[1]
        + x3[0] * x4[1]
        + x4[0] * x1[1]
        - x2[0] * x1[1]
        - x3[0] * x2[1]
        - x4[0] * x3[1]
        - x1[0] * x4[1]
    )


def reorder_corners(x1, x2, x3, x4):
    order = [0, 1, 3, 4]
    if triangle_area(x1, x4, x3) * triangle_area(x1, x2, x3) > 0:
        order = [0, 1, 4, 3]
    return order
