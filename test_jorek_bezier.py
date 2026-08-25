import unittest

import numpy as np

from jorek_bezier import edge_bezier_points, element_bezier_points


class EdgeBezierPointsTest(unittest.TestCase):
    def test_preserves_existing_control_point_formulas(self):
        points = np.arange(8, dtype=float).reshape(2, 2, 2)

        expected = np.empty((2, 2, 2))
        expected[:, 0, :] = points[:, 0, :]
        expected[:, 1, :] = points[:, 0, :] + points[:, 1, :]

        np.testing.assert_array_equal(edge_bezier_points(points), expected)


class ElementBezierPointsTest(unittest.TestCase):
    def test_preserves_existing_control_point_formulas_and_scaling(self):
        points = np.arange(32, dtype=float).reshape(2, 4, 4)
        scaling = 3.5

        expected = np.empty((2, 4, 4))
        expected[:, 0, 0] = points[:, 0, 0]
        expected[:, 1, 0] = points[:, 0, 0] + points[:, 1, 0]
        expected[:, 0, 1] = points[:, 0, 0] + points[:, 2, 0]
        expected[:, 1, 1] = points[:, :, 0].sum(axis=1)

        expected[:, 3, 0] = points[:, 0, 1]
        expected[:, 3, 1] = points[:, 0, 1] + points[:, 2, 1]
        expected[:, 2, 0] = points[:, 0, 1] + points[:, 1, 1]
        expected[:, 2, 1] = points[:, :, 1].sum(axis=1)

        expected[:, 3, 3] = points[:, 0, 2]
        expected[:, 3, 2] = points[:, 0, 2] + points[:, 2, 2]
        expected[:, 2, 3] = points[:, 0, 2] + points[:, 1, 2]
        expected[:, 2, 2] = points[:, :, 2].sum(axis=1)

        expected[:, 0, 3] = points[:, 0, 3]
        expected[:, 0, 2] = points[:, 0, 3] + points[:, 2, 3]
        expected[:, 1, 3] = points[:, 0, 3] + points[:, 1, 3]
        expected[:, 1, 2] = points[:, :, 3].sum(axis=1)

        np.testing.assert_array_equal(
            element_bezier_points(points, scaling), scaling * expected
        )


if __name__ == "__main__":
    unittest.main()
