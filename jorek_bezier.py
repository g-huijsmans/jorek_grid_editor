import numpy as np


def edge_bezier_points(points):
    """Return the cubic Bezier control points for one JOREK edge."""
    bezier_points = np.zeros((2, 2, 2))

    bezier_points[:, 0, 0] = points[:, 0, 0]
    bezier_points[:, 1, 0] = points[:, 0, 0] + points[:, 1, 0]
    bezier_points[:, 0, 1] = points[:, 0, 1]
    bezier_points[:, 1, 1] = points[:, 0, 1] + points[:, 1, 1]

    return bezier_points


def element_bezier_points(points, scaling):
    """Return the 4 by 4 Bezier control net for one JOREK element."""
    bezier_points = np.zeros((2, 4, 4))

    bezier_points[:, 0, 0] = points[:, 0, 0]
    bezier_points[:, 1, 0] = points[:, 0, 0] + points[:, 1, 0]
    bezier_points[:, 0, 1] = points[:, 0, 0] + points[:, 2, 0]
    bezier_points[:, 1, 1] = points[:, 0, 0] + points[:, 1, 0] + points[:, 2, 0] + points[:, 3, 0]

    bezier_points[:, 3, 0] = points[:, 0, 1]
    bezier_points[:, 3, 1] = points[:, 0, 1] + points[:, 2, 1]
    bezier_points[:, 2, 0] = points[:, 0, 1] + points[:, 1, 1]
    bezier_points[:, 2, 1] = points[:, 0, 1] + points[:, 1, 1] + points[:, 2, 1] + points[:, 3, 1]

    bezier_points[:, 3, 3] = points[:, 0, 2]
    bezier_points[:, 3, 2] = points[:, 0, 2] + points[:, 2, 2]
    bezier_points[:, 2, 3] = points[:, 0, 2] + points[:, 1, 2]
    bezier_points[:, 2, 2] = points[:, 0, 2] + points[:, 1, 2] + points[:, 2, 2] + points[:, 3, 2]

    bezier_points[:, 0, 3] = points[:, 0, 3]
    bezier_points[:, 0, 2] = points[:, 0, 3] + points[:, 2, 3]
    bezier_points[:, 1, 3] = points[:, 0, 3] + points[:, 1, 3]
    bezier_points[:, 1, 2] = points[:, 0, 3] + points[:, 1, 3] + points[:, 2, 3] + points[:, 3, 3]

    return scaling * bezier_points
