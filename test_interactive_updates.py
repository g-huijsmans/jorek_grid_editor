import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide2.QtCore import QPoint, QPointF, QRect, Qt
from PySide2.QtGui import QBrush, QColor, QPainterPath, QPen, QTransform
from PySide2.QtTest import QTest
from PySide2.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsScene,
)

import grid_editor5
from grid_editor5 import big_patch, big_patch_node, jorek_node_item, this_view
from jorek_bezier import edge_bezier_points, element_bezier_points


class UpdateRecorder:
    def __init__(self, uv_index=None):
        self.uv_index = uv_index
        self.update_count = 0

    def update(self):
        self.update_count += 1


def install_edge_owner_lookup(monkeypatch, edges):
    """Give lightweight boundary fixtures the owner metadata production has."""
    owners = {}
    for edge in edges:
        owner = owners.get(edge.element_index)
        if owner is None:
            owner = SimpleNamespace(
                index=edge.element_index,
                active=True,
                vertices=np.full(4, -1, dtype=int),
                sizes=np.ones((4, 4)),
            )
            owners[edge.element_index] = owner
        for endpoint, local_vertex in enumerate(edge.local_nodes_index):
            owner.vertices[local_vertex] = edge.vertices[endpoint]

    real_element_by_index = grid_editor5.element_by_index

    def element_by_index(index):
        return owners.get(index) or real_element_by_index(index)

    monkeypatch.setattr(grid_editor5, "element_by_index", element_by_index)
    return owners


def structured_editable_topology(rows=5, columns=5):
    node_columns = columns + 1
    elements = []
    boundary_edges = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            lower_left = row * node_columns + column
            vertices = [
                lower_left,
                lower_left + 1,
                lower_left + node_columns + 1,
                lower_left + node_columns,
            ]
            elements.append(SimpleNamespace(
                index=index, vertices=vertices, active=True
            ))
            if row in (0, rows - 1) or column in (0, columns - 1):
                boundary_edges.append(SimpleNamespace(
                    element_index=index, active=True
                ))
    return elements, boundary_edges


def editable_depth_grid(size=5):
    nodes_xx = np.zeros((2, 4, size * size))
    for row in range(size):
        for column in range(size):
            index = row * size + column
            nodes_xx[:, 0, index] = [column, row]
            nodes_xx[:, 1, index] = [1.0, 0.0]
            nodes_xx[:, 2, index] = [0.0, 1.0]
    boundary = np.array([
        int(row in (0, size - 1) or column in (0, size - 1))
        for row in range(size)
        for column in range(size)
    ], dtype=np.int32)
    vertices = []
    for row in range(size - 1):
        for column in range(size - 1):
            lower_left = row * size + column
            vertices.append([
                lower_left, lower_left + 1,
                lower_left + size + 1, lower_left + size,
            ])
    vertices = np.asarray(vertices, dtype=np.int32).T
    element_sizes = np.zeros((4, 4, vertices.shape[1]))
    element_sizes[0, :, :] = 1.0
    element_sizes[1, :, :] = np.array([1, -1, -1, 1])[:, None] / 3.0
    element_sizes[2, :, :] = np.array([1, 1, -1, -1])[:, None] / 3.0
    return SimpleNamespace(
        nodes_xx=nodes_xx,
        boundary=boundary,
        vertices=vertices,
        elements_size=element_sizes,
    )


def rubberband_selection_case(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, 4))
    nodes_xx[0, 0, :] = np.arange(4.0)
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 1)
        for index in range(4)
    ]
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx),
        raising=False,
    )
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    elements = [
        grid_editor5.jorek_element_item(
            index, np.array([0, 1, 2, 3]), np.ones((4, 4))
        )
        for index in range(3)
    ]
    edges = [
        grid_editor5.boundary_edge(
            nodes[index:index + 2], [index, index + 1], [0, 1],
            index, 0, 1, np.ones((2, 2)),
        )
        for index in range(3)
    ]
    for item in nodes + elements + edges:
        scene.addItem(item)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", elements, raising=False)
    monkeypatch.setattr(grid_editor5, "boundary_list", edges, raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)
    return SimpleNamespace(
        app=app, scene=scene, view=view, nodes=nodes,
        elements=elements, edges=edges,
    )


def perform_rubberband_selection(view, monkeypatch, modifiers, hit_edges):
    monkeypatch.setattr(
        view, "items",
        lambda position: list(hit_edges) if isinstance(position, QRect) else [],
    )

    def event(position, event_modifiers=modifiers):
        return SimpleNamespace(
            pos=lambda: position,
            modifiers=lambda: event_modifiers,
            accept=lambda: None,
        )

    view.mousePressEvent(event(QPoint(10, 10)))
    view.mouseMoveEvent(event(QPoint(20, 20)))
    mode = view.rubberband_mode
    view.mouseReleaseEvent(event(QPoint(20, 20), Qt.NoModifier))
    return mode


def assert_basis_handles_match_vectors(node):
    for basis_index, handle in (
        (1, node.blue_handle),
        (2, node.red_handle),
    ):
        expected = node.position + grid_editor5.qt_point(
            grid_editor5.node_basis_display_vector(node, basis_index)
        )
        assert handle.pos() == expected


def test_element_and_boundary_point_scaling_does_not_modify_grid(monkeypatch):
    app = QApplication.instance() or QApplication([])
    nodes_xx = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    vertices = np.array([0, 1, 2, 3])
    sizes = np.linspace(0.25, 1.75, 16).reshape(4, 4)
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index].copy(), 1)
        for index in range(4)
    ]
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx),
        raising=False,
    )
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    before = nodes_xx.copy()

    element = grid_editor5.jorek_element_item(0, vertices, sizes)
    edges = element.find_boundary_edges()

    assert np.array_equal(nodes_xx, before)
    assert not np.shares_memory(element.points, nodes_xx)
    assert edges
    assert all(not np.shares_memory(edge.points, nodes_xx) for edge in edges)
    assert app is not None


def display_element(index, node_index, scale_u, scale_v):
    sizes = np.ones((4, 1))
    sizes[1, 0] = scale_u
    sizes[2, 0] = scale_v
    return SimpleNamespace(
        index=index, vertices=np.array([node_index]), sizes=sizes,
        active=True, update=lambda: None,
    )


def display_boundary_edge(element_index, node_index, uv_index=1):
    return SimpleNamespace(
        element_index=element_index, vertices=[node_index], uv_index=uv_index,
        active=True, update=lambda: None,
    )


def test_boundary_display_reference_is_owner_preferred_and_deterministic():
    node = jorek_node_item(0, np.zeros((2, 4)), 1)
    lower_element = display_element(3, node.index, 0.2, -0.4)
    boundary_owner = display_element(9, node.index, 0.6, -0.8)
    node.connected_elements = [boundary_owner, lower_element]
    node.connected_boundary_edges = [
        display_boundary_edge(boundary_owner.index, node.index)
    ]

    reference = grid_editor5.node_display_reference(node)
    assert reference[0] is boundary_owner and reference[1] == 0
    assert grid_editor5.node_basis_display_scale(node, 1) == pytest.approx(0.6)
    assert grid_editor5.node_basis_display_scale(node, 2) == pytest.approx(-0.8)

    node.connected_boundary_edges.append(
        display_boundary_edge(lower_element.index, node.index)
    )
    reference = grid_editor5.node_display_reference(node)
    assert reference[0] is lower_element and reference[1] == 0


def test_display_vectors_and_handles_use_element_local_scales():
    node_xx = np.zeros((2, 4))
    node_xx[:, 0] = [10.0, 20.0]
    node_xx[:, 1] = [6.0, -2.0]
    node_xx[:, 2] = [-3.0, 5.0]
    node = jorek_node_item(0, node_xx, 1)
    element = display_element(4, node.index, 0.25, -0.5)
    node.connected_elements = [element]
    node.connected_boundary_edges = [
        display_boundary_edge(element.index, node.index)
    ]

    node.blue_handle.sync_position()
    node.red_handle.sync_position()

    assert np.allclose(
        grid_editor5.node_basis_display_vector(node, 1), [1.5, -0.5]
    )
    assert np.allclose(
        grid_editor5.node_basis_display_vector(node, 2), [1.5, -2.5]
    )
    assert_basis_handles_match_vectors(node)


@pytest.mark.parametrize(
    "basis_index, reference_scale, effective_scene_vector",
    [(1, 0.25, np.array([20.0, -30.0])),
     (2, -0.5, np.array([20.0, -30.0]))],
)
def test_handle_drag_stores_raw_basis_using_signed_reference_scale(
    monkeypatch, basis_index, reference_scale, effective_scene_vector
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, 1))
    node_xx = np.zeros((2, 4))
    node_xx[:, 0] = [100.0, 200.0]
    node = jorek_node_item(0, node_xx, 1)
    scene.addItem(node)
    scale_u = reference_scale if basis_index == 1 else 0.75
    scale_v = reference_scale if basis_index == 2 else 0.75
    element = display_element(7, node.index, scale_u, scale_v)
    node.connected_elements = [element]
    node.connected_boundary_edges = [
        display_boundary_edge(element.index, node.index, basis_index)
    ]
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx),
        raising=False,
    )
    monkeypatch.setattr(grid_editor5, "this_scaling", 10.0, raising=False)
    handle = node.blue_handle if basis_index == 1 else node.red_handle
    target = node.position + grid_editor5.qt_point(effective_scene_vector)

    handle.move_to_scene(target)

    effective_physical = effective_scene_vector / 10.0
    expected_raw = effective_physical / reference_scale
    assert np.allclose(nodes_xx[:, basis_index, node.index], expected_raw)
    assert np.allclose(node.xx[:, basis_index], 10.0 * expected_raw)
    assert np.allclose(reference_scale * expected_raw, effective_physical)
    assert handle.pos() == target
    assert app is not None


def test_display_reference_reselects_after_owner_becomes_inactive():
    node = jorek_node_item(0, np.zeros((2, 4)), 1)
    first = display_element(1, node.index, 0.2, 0.3)
    second = display_element(2, node.index, 0.4, 0.5)
    node.connected_elements = [second, first]
    node.connected_boundary_edges = [
        display_boundary_edge(first.index, node.index),
        display_boundary_edge(second.index, node.index),
    ]

    reference = grid_editor5.node_display_reference(node)
    assert reference[0] is first and reference[1] == 0
    first.active = False
    reference = grid_editor5.node_display_reference(node)
    assert reference[0] is second and reference[1] == 0
    assert grid_editor5.node_basis_display_scale(node, 1) == pytest.approx(0.4)


def test_bezier_nodal_parameter_scales_use_adjacent_interval_means():
    scales = grid_editor5.bezier_nodal_parameter_scales(
        [0.0, 0.2, 0.5, 1.0]
    )
    assert np.allclose(scales, [0.2, 0.25, 0.4, 0.5])


@pytest.mark.parametrize(
    "parameters",
    [[], [0.0], [0.0, 0.0], [0.0, -0.1], [0.0, np.nan]],
)
def test_bezier_nodal_parameter_scales_reject_invalid_intervals(parameters):
    with pytest.raises(ValueError, match="positive|at least two"):
        grid_editor5.bezier_nodal_parameter_scales(parameters)


def test_prescribed_bezier_sizes_preserve_parameter_order_when_reversed():
    parameter_start_node = object()
    parameter_end_node = object()
    sizes = grid_editor5.prescribed_bezier_outer_edge_sizes(
        [parameter_end_node, parameter_start_node],
        parameter_start_node,
        parameter_end_node,
        parameter_interval=0.3,
        parameter_start_scale=0.2,
        parameter_end_scale=0.4,
    )
    assert np.allclose(sizes[1, :], [-0.75, 1.5])


def test_orient_new_node_red_vector_synchronizes_flipped_handle(monkeypatch):
    app = QApplication.instance() or QApplication([])
    reference_xx = np.zeros((2, 4))
    reference_xx[:, 2] = [0.0, 1.0]
    new_xx = np.zeros((2, 4))
    new_xx[:, 2] = [0.0, -2.0]
    reference_node = jorek_node_item(0, reference_xx, 2)
    new_node = jorek_node_item(1, new_xx, 2)
    nodes_xx = np.stack((reference_xx, new_xx), axis=2)
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx), raising=False
    )

    old_handle_position = QPointF(new_node.red_handle.pos())
    grid_editor5.orient_new_node_red_vector(
        new_node, reference_node, column=0, main_uv_index=1, perp_index=2
    )

    assert new_node.red_handle.pos() != old_handle_position
    assert_basis_handles_match_vectors(new_node)
    assert np.array_equal(
        new_node.xx[:, 2], grid_editor5.jorek.nodes_xx[:, 2, new_node.index]
    )
    assert app is not None


def test_node_drag_keeps_press_target_when_nodes_overlap(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, 2))
    nodes_xx[:, 0, 1] = [100.0, 0.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 0)
        for index in range(2)
    ]
    for node in nodes:
        scene.addItem(node)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx), raising=False
    )
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(
        grid_editor5.QApplication, "keyboardModifiers", lambda: Qt.NoModifier
    )

    press_position = QPoint(10, 10)
    press_event = SimpleNamespace(
        pos=lambda: press_position,
        accept=lambda: None,
    )
    monkeypatch.setattr(view, "items", lambda position: [nodes[0]])
    view.mousePressEvent(press_event)
    assert view.dragged_node is nodes[0]
    assert view.rubberBand is None

    move_position = QPoint(100, 50)
    expected_position = view.mapToScene(move_position)
    original_second_position = QPointF(nodes[1].position)
    monkeypatch.setattr(
        view,
        "items",
        lambda position: pytest.fail("node drag must not repeat hit-testing"),
    )
    move_event = SimpleNamespace(
        pos=lambda: move_position,
        accept=lambda: None,
    )
    view.mouseMoveEvent(move_event)

    assert nodes[0].position == expected_position
    assert nodes[1].position == original_second_position
    assert_basis_handles_match_vectors(nodes[0])

    release_event = SimpleNamespace(accept=lambda: None)
    view.mouseReleaseEvent(release_event)
    assert view.dragged_node is None
    assert app is not None


def geometry_undo_case(monkeypatch, node_count=2):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, node_count))
    for index in range(node_count):
        nodes_xx[:, 0, index] = [float(2 * index), float(index)]
        nodes_xx[:, 1, index] = [0.4 + index, 0.1]
        nodes_xx[:, 2, index] = [0.2, 0.5 + index]
    nodes = [
        jorek_node_item(index, np.array(nodes_xx[:, :, index], copy=True), 2)
        for index in range(node_count)
    ]
    for node in nodes:
        scene.addItem(node)
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx), raising=False
    )
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(grid_editor5, "boundary_list", [], raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)
    monkeypatch.setattr(grid_editor5, "static_mesh", None, raising=False)
    return SimpleNamespace(
        app=app, scene=scene, view=view, nodes=nodes, nodes_xx=nodes_xx
    )


def ctrl_z_event():
    return SimpleNamespace(
        key=lambda: Qt.Key_Z,
        modifiers=lambda: Qt.ControlModifier,
    )


def complete_node_drag(case, node, position):
    case.view.begin_geometry_drag(node, "node")
    case.view.dragged_node = node
    node.move_to_scene(position)
    case.view.mouseReleaseEvent(SimpleNamespace(accept=lambda: None))


def complete_basis_drag(case, handle, position):
    case.view.begin_geometry_drag(handle.node, "basis-vector")
    case.view.selected_point = handle
    handle.move_to_scene(position)
    case.view.mouseReleaseEvent(SimpleNamespace(accept=lambda: None))


def test_ctrl_z_restores_complete_node_state_and_connected_geometry(monkeypatch):
    case = geometry_undo_case(monkeypatch)
    node = case.nodes[0]
    element = UpdateRecorder()
    element.index = 10
    element.vertices = [0]
    element.sizes = np.ones((4, 1))
    element.active = True
    boundary = UpdateRecorder(uv_index=2)
    boundary.element_index = 10
    boundary.vertices = [0]
    boundary.active = True
    node.connected_elements = [element]
    node.connected_boundary_edges = [boundary]
    static_refreshes = []
    monkeypatch.setattr(
        grid_editor5, "rebuild_static_mesh_path",
        lambda expected_scene=None: static_refreshes.append(expected_scene),
    )
    original = np.array(case.nodes_xx[:, :, 0], copy=True)

    complete_node_drag(case, node, QPointF(7.5, -3.25))
    assert case.view.last_geometry_undo is not None
    case.view.keyPressEvent(ctrl_z_event())

    assert np.array_equal(case.nodes_xx[:, :, 0], original)
    assert np.array_equal(node.xx, original)
    assert node.position == QPointF(*original[:, 0])
    assert node.ellipse_item.pos() == node.position
    assert_basis_handles_match_vectors(node)
    assert element.update_count >= 2
    assert boundary.update_count >= 2
    assert len(static_refreshes) == 2
    assert case.view.last_geometry_undo is None


@pytest.mark.parametrize("basis_index", [1, 2])
def test_ctrl_z_restores_raw_basis_and_display_handle(monkeypatch, basis_index):
    case = geometry_undo_case(monkeypatch)
    node = case.nodes[0]
    handle = node.blue_handle if basis_index == 1 else node.red_handle
    original = np.array(case.nodes_xx[:, :, 0], copy=True)
    original_handle_position = QPointF(handle.pos())

    complete_basis_drag(case, handle, node.position + QPointF(1.7, -0.8))
    assert not np.array_equal(case.nodes_xx[:, basis_index, 0], original[:, basis_index])
    case.view.keyPressEvent(ctrl_z_event())

    assert np.array_equal(case.nodes_xx[:, :, 0], original)
    assert np.array_equal(node.xx[:, basis_index], original[:, basis_index])
    assert handle.pos() == original_handle_position
    assert_basis_handles_match_vectors(node)


def test_geometry_undo_is_single_level_and_consumed(monkeypatch):
    case = geometry_undo_case(monkeypatch)
    node0, node1 = case.nodes
    complete_node_drag(case, node0, QPointF(5.0, 5.0))
    moved_node0 = np.array(case.nodes_xx[:, :, 0], copy=True)
    original_node1 = np.array(case.nodes_xx[:, :, 1], copy=True)
    complete_node_drag(case, node1, QPointF(8.0, 9.0))

    case.view.keyPressEvent(ctrl_z_event())
    assert np.array_equal(case.nodes_xx[:, :, 0], moved_node0)
    assert np.array_equal(case.nodes_xx[:, :, 1], original_node1)
    after_first_undo = np.array(case.nodes_xx, copy=True)
    case.view.keyPressEvent(ctrl_z_event())
    assert np.array_equal(case.nodes_xx, after_first_undo)


def test_noop_geometry_drag_preserves_previous_undo(monkeypatch):
    case = geometry_undo_case(monkeypatch)
    node0, node1 = case.nodes
    original_node0 = np.array(case.nodes_xx[:, :, 0], copy=True)
    complete_node_drag(case, node0, QPointF(5.0, 5.0))
    saved_undo = case.view.last_geometry_undo

    case.view.begin_geometry_drag(node1, "node")
    case.view.dragged_node = node1
    case.view.mouseReleaseEvent(SimpleNamespace(accept=lambda: None))

    assert case.view.last_geometry_undo is saved_undo
    case.view.keyPressEvent(ctrl_z_event())
    assert np.array_equal(case.nodes_xx[:, :, 0], original_node0)


def test_geometry_undo_uses_node_index_after_overlay_item_replacement(monkeypatch):
    case = geometry_undo_case(monkeypatch, node_count=1)
    original = np.array(case.nodes_xx[:, :, 0], copy=True)
    complete_node_drag(case, case.nodes[0], QPointF(4.0, 6.0))
    case.scene.removeItem(case.nodes[0])
    replacement = jorek_node_item(
        0, np.array(case.nodes_xx[:, :, 0], copy=True), 2
    )
    case.scene.addItem(replacement)
    grid_editor5.node_list[0] = replacement

    case.view.keyPressEvent(ctrl_z_event())

    assert np.array_equal(case.nodes_xx[:, :, 0], original)
    assert np.array_equal(replacement.xx, original)
    assert replacement.position == QPointF(*original[:, 0])
    assert_basis_handles_match_vectors(replacement)


def test_ctrl_z_is_ignored_during_active_geometry_drag(monkeypatch):
    case = geometry_undo_case(monkeypatch, node_count=1)
    node = case.nodes[0]
    original = np.array(case.nodes_xx[:, :, 0], copy=True)
    complete_node_drag(case, node, QPointF(3.0, 4.0))
    case.view.begin_geometry_drag(node, "node")
    case.view.dragged_node = node

    case.view.keyPressEvent(ctrl_z_event())

    assert not np.array_equal(case.nodes_xx[:, :, 0], original)
    assert case.view.last_geometry_undo is not None


@pytest.mark.parametrize(
    "start,end",
    [
        (QPoint(20, 30), QPoint(80, 90)),
        (QPoint(80, 90), QPoint(20, 30)),
        (QPoint(80, 30), QPoint(20, 90)),
        (QPoint(20, 90), QPoint(80, 30)),
    ],
)
def test_visible_rubber_band_is_normalized_in_every_drag_direction(
    monkeypatch, start, end
):
    app = QApplication.instance() or QApplication([])
    view = this_view()
    view.resize(320, 240)
    view.setScene(QGraphicsScene())
    monkeypatch.setattr(
        grid_editor5.QApplication, "keyboardModifiers", lambda: Qt.NoModifier
    )

    def event(position):
        return SimpleNamespace(pos=lambda: position, accept=lambda: None)

    view.mousePressEvent(event(start))
    view.mouseMoveEvent(event(end))

    expected = QRect(start, end).normalized()
    geometry = view.rubberBand.geometry()
    assert geometry.width() >= 0
    assert geometry.height() >= 0
    assert geometry == expected
    assert view.end_point == end
    assert app is not None


def test_rubber_band_and_item_query_share_viewport_coordinates(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.auto_fit_on_resize = False
    view.resize(320, 240)
    view.setScene(scene)
    view.show()
    app.processEvents()

    line_items = []
    for screen_x in (97, 100, 103):
        path = QPainterPath()
        path.moveTo(view.mapToScene(QPoint(screen_x, 80)))
        path.lineTo(view.mapToScene(QPoint(screen_x, 120)))
        item = QGraphicsPathItem(path)
        pen = QPen(Qt.yellow)
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        item.setPen(pen)
        scene.addItem(item)
        line_items.append(item)

    monkeypatch.setattr(
        grid_editor5.QApplication, "keyboardModifiers", lambda: Qt.NoModifier
    )
    start = QPoint(100, 90)
    end = QPoint(101, 110)

    def event(position):
        return SimpleNamespace(pos=lambda: position, accept=lambda: None)

    view.mousePressEvent(event(start))
    view.mouseMoveEvent(event(end))

    query_rect = QRect(start, end).normalized()
    band_geometry = view.rubberBand.geometry()
    visible_rect = QRect(
        view.viewport().mapFrom(
            view.rubberBand.parentWidget(), band_geometry.topLeft()
        ),
        view.viewport().mapFrom(
            view.rubberBand.parentWidget(), band_geometry.bottomRight()
        ),
    ).normalized()
    returned_lines = [
        item for item in view.items(query_rect) if item in line_items
    ]

    assert view.frameWidth() == 1
    assert view.viewport().pos() == QPoint(1, 1)
    assert view.rubberBand.parentWidget() is view.viewport()
    assert band_geometry == query_rect
    assert visible_rect == query_rect
    assert returned_lines == [line_items[1]]
    assert app is not None


def test_cosmetic_boundary_edge_shape_matches_visible_screen_width(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.resize(320, 240)
    view.setScene(scene)
    view.zoom_level = 10.0
    view.setTransform(QTransform().scale(10.0, 10.0))
    view.setSceneRect(-1.0, -1.0, 5.0, 2.0)
    view.show()
    app.processEvents()

    nodes_xx = np.zeros((2, 4, 4))
    nodes_xx[0, 0, :] = np.arange(4.0)
    nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(4)
    ]
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx),
        raising=False,
    )
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)

    edges = []
    for index in range(3):
        edge = grid_editor5.boundary_edge(
            nodes[index:index + 2], [index, index + 1], [0, 1],
            index, 0, 1, np.ones((2, 2)),
        )
        path = QPainterPath(QPointF(float(index), 0.0))
        path.lineTo(QPointF(float(index + 1), 0.0))
        edge.setPath(path)
        scene.addItem(edge)
        edges.append(edge)

    middle = view.mapFromScene(QPointF(1.5, 0.0))
    query_rect = QRect(middle.x() - 1, middle.y() - 1, 3, 3)
    returned_edges = [
        item for item in view.items(query_rect) if item in edges
    ]
    screen_shape = edges[1].deviceTransform(
        view.viewportTransform()
    ).map(edges[1].shape())

    assert returned_edges == [edges[1]]
    assert screen_shape.boundingRect().height() == pytest.approx(
        grid_editor5.BOUNDARY_EDGE_WIDTH
    )
    assert app is not None


def test_boundary_edge_normal_pen_is_cyan_and_selected_pen_remains_green():
    normal_pen = grid_editor5.boundary_edge_pen()
    selected_pen = grid_editor5.boundary_edge_pen(Qt.green, 2.5)

    assert normal_pen.color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert normal_pen.widthF() == pytest.approx(
        grid_editor5.BOUNDARY_EDGE_WIDTH
    )
    assert selected_pen.color() == Qt.green
    assert selected_pen.widthF() == pytest.approx(2.5)


def test_shift_rubberband_adds_edges_without_clearing_or_duplicates(monkeypatch):
    case = rubberband_selection_case(monkeypatch)

    mode = perform_rubberband_selection(
        case.view, monkeypatch, Qt.ShiftModifier,
        [case.edges[0], case.edges[0]],
    )
    assert mode == "add"
    assert case.view.selected_edges == [case.edges[0]]
    assert case.edges[0].pen().color() == Qt.green

    perform_rubberband_selection(
        case.view, monkeypatch, Qt.ShiftModifier, [case.edges[2]]
    )
    assert case.view.selected_edges == [case.edges[0], case.edges[2]]
    assert case.edges[0].pen().color() == Qt.green
    assert case.edges[2].pen().color() == Qt.green

    perform_rubberband_selection(
        case.view, monkeypatch, Qt.ShiftModifier,
        [case.edges[0], case.edges[1], case.edges[1]],
    )
    assert case.view.selected_edges == [
        case.edges[0], case.edges[2], case.edges[1]
    ]
    assert len(case.view.selected_edges) == len(set(case.view.selected_edges))
    assert all(edge.pen().color() == Qt.green for edge in case.edges)
    assert all(edge.pen().widthF() == pytest.approx(2.5) for edge in case.edges)
    assert case.view.selected_nodes == case.nodes
    assert case.view.selected_elements == [
        case.elements[0], case.elements[2], case.elements[1]
    ]
    assert case.view.rubberBand is None
    assert case.view.start_point is None
    assert case.view.end_point is None
    assert case.view.rubberband_mode is None
    assert case.app is not None


def test_ctrl_shift_rubberband_toggles_mixed_edges_and_is_reversible(
    monkeypatch,
):
    case = rubberband_selection_case(monkeypatch)
    case.view.replace_boundary_edge_selection([case.edges[0]])

    mode = perform_rubberband_selection(
        case.view, monkeypatch,
        Qt.ControlModifier | Qt.ShiftModifier,
        [case.edges[0], case.edges[1], case.edges[1]],
    )

    assert mode == "toggle"
    assert case.view.selected_edges == [case.edges[1]]
    assert case.edges[0].pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert case.edges[1].pen().color() == Qt.green
    assert case.edges[2].pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert case.view.selected_nodes == [case.nodes[1], case.nodes[2]]
    assert case.view.selected_elements == [case.elements[1]]

    perform_rubberband_selection(
        case.view, monkeypatch,
        Qt.ControlModifier | Qt.ShiftModifier,
        [case.edges[0], case.edges[1]],
    )
    assert case.view.selected_edges == [case.edges[0]]
    assert case.edges[0].pen().color() == Qt.green
    assert case.edges[1].pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert case.edges[2].pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert case.view.selected_nodes == [case.nodes[0], case.nodes[1]]
    assert case.view.selected_elements == [case.elements[0]]

    perform_rubberband_selection(
        case.view, monkeypatch,
        Qt.ControlModifier | Qt.ShiftModifier,
        [case.edges[2]],
    )
    assert case.view.selected_edges == [case.edges[0], case.edges[2]]
    assert case.view.selected_elements == [case.elements[0], case.elements[2]]
    assert case.app is not None


def test_escape_clears_toggle_selection_and_pending_rubberband(monkeypatch):
    case = rubberband_selection_case(monkeypatch)
    case.view.replace_boundary_edge_selection([case.edges[0], case.edges[1]])
    monkeypatch.setattr(case.view, "items", lambda position: [])
    event = SimpleNamespace(
        pos=lambda: QPoint(10, 10),
        modifiers=lambda: Qt.ControlModifier | Qt.ShiftModifier,
        accept=lambda: None,
    )
    case.view.mousePressEvent(event)
    assert case.view.rubberband_mode == "toggle"

    case.view.keyPressEvent(SimpleNamespace(key=lambda: Qt.Key_Escape))

    assert case.view.selected_edges == []
    assert case.view.selected_nodes == []
    assert case.view.selected_elements == []
    assert case.view.rubberBand is None
    assert case.view.rubberband_mode is None
    assert all(
        edge.pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
        for edge in case.edges
    )
    assert case.app is not None


def test_resize_fits_only_active_grid_with_uniform_transform(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, 3))
    nodes_xx[:, 0, 0] = [0.0, 0.0]
    nodes_xx[:, 0, 1] = [100.0, 50.0]
    nodes_xx[:, 0, 2] = [10000.0, 10000.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 0)
        for index in range(3)
    ]
    nodes[2].active = False
    nodes[2].setVisible(False)
    for node in nodes:
        scene.addItem(node)
    preview = QGraphicsEllipseItem(20000.0, 20000.0, 100.0, 100.0)
    scene.addItem(preview)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)

    original_positions = [QPointF(node.position) for node in nodes]
    view.setTransform(QTransform().scale(1.0, -1.0))
    view.resize(600, 300)
    view.show()
    app.processEvents()
    view.fit_grid_to_window()

    grid_rect = view.grid_bounding_rect()
    transform = view.transform()
    assert view.auto_fit_on_resize
    assert grid_rect == grid_rect.normalized()
    assert grid_rect.left() == pytest.approx(0.0)
    assert grid_rect.right() == pytest.approx(100.0)
    assert grid_rect.bottom() == pytest.approx(50.0)
    assert abs(transform.m11()) == pytest.approx(abs(transform.m22()))
    assert transform.m22() < 0.0
    assert view.zoom_level == pytest.approx(
        np.hypot(transform.m11(), transform.m12())
    )
    assert [node.position for node in nodes] == original_positions
    view.close()


def test_fit_centers_combined_grid_and_wall_without_resize_drift(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, 2))
    nodes_xx[:, 0, 0] = [0.0, 0.0]
    nodes_xx[:, 0, 1] = [100.0, 40.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 0)
        for index in range(2)
    ]
    for node in nodes:
        scene.addItem(node)

    wall_path = QPainterPath(QPointF(-10.0, -20.0))
    wall_path.lineTo(QPointF(120.0, 60.0))
    wall = QGraphicsPathItem(wall_path)
    scene.addItem(wall)
    view.wall_outline_item = wall

    # Model non-fit graphics (for example editing handles) that make the
    # scene rectangle asymmetric while remaining small enough that Qt has no
    # scrollbar range after fitting.
    scene.addItem(QGraphicsEllipseItem(-15.0, -25.0, 1.0, 1.0))
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)

    view.resize(600, 400)
    view.show()
    app.processEvents()
    fit_center = view.grid_bounding_rect().center()
    for size in ((600, 400), (800, 500), (500, 700), (600, 400)):
        view.resize(*size)
        app.processEvents()
        view.fit_grid_to_window()
        app.processEvents()
        mapped_center = view.mapToScene(view.viewport().rect().center())
        transform = view.transform()
        horizontal_error = abs(
            mapped_center.x() - fit_center.x()
        ) * abs(transform.m11())
        vertical_error = abs(
            mapped_center.y() - fit_center.y()
        ) * abs(transform.m22())
        assert horizontal_error <= 2.0
        assert vertical_error <= 2.0
        assert transform.m11() > 0.0
        assert transform.m22() < 0.0
        assert view.auto_fit_on_resize
    view.close()


def test_resize_preserves_manual_view_transform_when_auto_fit_is_disabled():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene(-1000.0, -1000.0, 2000.0, 2000.0)
    view = this_view()
    view.setScene(scene)
    view.resize(300, 200)
    view.show()
    app.processEvents()

    view.setTransform(QTransform().scale(2.0, -2.0))
    view.zoom_level = 2.0
    view.centerOn(QPointF(125.0, -75.0))
    view.auto_fit_on_resize = True
    view.scrollContentsBy(1, 0)
    assert not view.auto_fit_on_resize
    app.processEvents()
    old_transform = QTransform(view.transform())

    view.resize(600, 400)
    app.processEvents()

    assert view.transform() == old_transform
    assert view.zoom_level == pytest.approx(2.0)
    assert not view.auto_fit_on_resize
    view.close()


def test_resize_calls_fit_path_when_auto_fit_is_enabled(monkeypatch):
    app = QApplication.instance() or QApplication([])
    view = this_view()
    view.setScene(QGraphicsScene())
    view.resize(300, 200)
    view.show()
    app.processEvents()
    calls = []
    real_fit = view.fit_grid_to_window

    def record_fit():
        calls.append(view.viewport().size())
        real_fit()

    monkeypatch.setattr(view, "fit_grid_to_window", record_fit)
    view.auto_fit_on_resize = True
    view.resize(600, 400)
    app.processEvents()

    assert calls
    assert view.auto_fit_on_resize
    view.close()


def test_f_key_fits_active_grid_after_manual_zoom_and_pan(
    monkeypatch, capsys
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene(-5000.0, -5000.0, 10000.0, 10000.0)
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, 2))
    nodes_xx[:, 0, 0] = [0.0, 0.0]
    nodes_xx[:, 0, 1] = [100.0, 50.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 0)
        for index in range(2)
    ]
    for node in nodes:
        scene.addItem(node)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    view.resize(600, 300)
    view.show()
    app.processEvents()

    view.setTransform(QTransform().scale(4.0, -4.0))
    view.zoom_level = 4.0
    view.centerOn(QPointF(1000.0, 1000.0))
    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_F})()
    view.keyPressEvent(event)

    transform = view.transform()
    viewport_rect = view.viewport().rect()
    mapped_corners = [
        view.mapFromScene(QPointF(x, y))
        for x, y in ((0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0))
    ]
    assert all(viewport_rect.contains(point) for point in mapped_corners)
    assert abs(transform.m11()) == pytest.approx(abs(transform.m22()))
    assert transform.m22() < 0.0
    assert view.zoom_level == pytest.approx(
        np.hypot(transform.m11(), transform.m12())
    )
    assert view.auto_fit_on_resize
    assert "fit grid to window" in capsys.readouterr().out
    view.close()


def test_extended_patch_number_key_rebuilds_radial_preview_rows(
    monkeypatch, capsys
):
    app = QApplication.instance() or QApplication([])
    nodes_xx = np.zeros((2, 4, 3))
    nodes_xx[:, 0, :] = [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 1)
        for index in range(3)
    ]
    edges = [
        SimpleNamespace(
            nodes=[nodes[index], nodes[index + 1]],
            vertices=[index, index + 1], uv_index=1,
            element_index=100 + index,
        )
        for index in range(2)
    ]
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    patch = grid_editor5.extended_patch(nodes, edges)
    patch.set_outer_positions([
        QPointF(0.0, 4.0), QPointF(1.0, 5.0), QPointF(2.0, 4.0)
    ])
    outer_nodes = list(patch.outer_nodes)
    view = this_view()
    view.current_extended_patch = patch
    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_4})()

    view.keyPressEvent(event)

    assert patch.radial_layers == 4
    assert view.pending_main_uv_index is None
    assert len(patch.preview_node_rows) == 5
    assert patch.preview_node_rows[0] is patch.ordered_nodes
    assert patch.preview_node_rows[-1] is patch.outer_nodes
    assert patch.outer_nodes == outer_nodes
    for column, (inner_node, outer_node) in enumerate(zip(nodes, outer_nodes)):
        for radial_index in range(1, 4):
            fraction = radial_index / 4.0
            position = patch.preview_node_rows[radial_index][column].position
            assert position.x() == pytest.approx(
                (1.0 - fraction) * inner_node.position.x()
                + fraction * outer_node.position.x()
            )
            assert position.y() == pytest.approx(
                (1.0 - fraction) * inner_node.position.y()
                + fraction * outer_node.position.y()
            )
    assert "extended patch radial layers: 4" in capsys.readouterr().out
    assert app is not None

    patch.enable_bezier_mode()
    bezier_outer_nodes = list(patch.outer_nodes)
    bezier_positions = [QPointF(node.position) for node in patch.outer_nodes]
    bezier_tangents = np.array(patch.outer_tangents)
    patch.set_radial_layers(3)

    assert patch.radial_layers == 3
    assert len(patch.preview_node_rows) == 4
    assert patch.outer_nodes == bezier_outer_nodes
    assert [node.position for node in patch.outer_nodes] == bezier_positions
    assert np.array_equal(np.array(patch.outer_tangents), bezier_tangents)


def regular_automatic_extended_patch(monkeypatch):
    positions = [
        (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
    ]
    nodes = [
        SimpleNamespace(index=index, position=QPointF(*position))
        for index, position in enumerate(positions)
    ]
    edges = [
        SimpleNamespace(
            nodes=[nodes[0], nodes[1]], vertices=[0, 1],
            uv_index=1, element_index=10,
        ),
        SimpleNamespace(
            nodes=[nodes[1], nodes[2]], vertices=[1, 2],
            uv_index=1, element_index=11,
        ),
    ]
    elements = [
        SimpleNamespace(index=10, vertices=[3, 4, 1, 0]),
        SimpleNamespace(index=11, vertices=[4, 5, 2, 1]),
    ]
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", elements, raising=False)
    return grid_editor5.extended_patch(nodes[:3], edges)


@pytest.mark.parametrize(
    "radial_layers", [1, 2, 3, 4, 10, 20, grid_editor5.MAX_RADIAL_LAYERS]
)
def test_automatic_zero_cap_extent_is_one_base_width_per_layer(
    monkeypatch, radial_layers
):
    patch = regular_automatic_extended_patch(monkeypatch)
    patch.set_radial_layers(radial_layers)
    patch.initialize_automatic_outer_geometry()

    assert np.allclose(patch.base_radial_displacement, [0.0, 1.0])
    for inner_node, outer_node in zip(patch.ordered_nodes, patch.outer_nodes):
        assert np.allclose(
            grid_editor5.np_point(outer_node.position)
            - grid_editor5.np_point(inner_node.position),
            [0.0, float(radial_layers)],
        )
    for column in range(len(patch.ordered_nodes)):
        row_y = [row[column].position.y() for row in patch.preview_node_rows]
        assert np.diff(row_y) == pytest.approx(np.ones(radial_layers))


def test_radial_layer_limit_rejects_only_values_above_central_max(monkeypatch):
    patch = regular_automatic_extended_patch(monkeypatch)
    patch.set_radial_layers(grid_editor5.MAX_RADIAL_LAYERS)
    assert patch.radial_layers == grid_editor5.MAX_RADIAL_LAYERS

    with pytest.raises(ValueError, match=str(grid_editor5.MAX_RADIAL_LAYERS)):
        patch.set_radial_layers(grid_editor5.MAX_RADIAL_LAYERS + 1)

    view = this_view()
    with pytest.raises(ValueError, match=str(grid_editor5.MAX_RADIAL_LAYERS)):
        view.set_extended_radial_layers(grid_editor5.MAX_RADIAL_LAYERS + 1)


def test_automatic_zero_cap_layer_changes_recompute_from_base_width(monkeypatch):
    patch = regular_automatic_extended_patch(monkeypatch)
    patch.initialize_automatic_outer_geometry()

    patch.set_radial_layers(3)
    assert patch.outer_nodes[0].position.y() == pytest.approx(4.0)
    patch.set_radial_layers(2)
    assert patch.outer_nodes[0].position.y() == pytest.approx(3.0)
    patch.set_radial_layers(4)
    assert patch.outer_nodes[0].position.y() == pytest.approx(5.0)
    assert np.allclose(patch.base_radial_displacement, [0.0, 1.0])


def test_automatic_zero_cap_bezier_preserves_scaled_extent(monkeypatch):
    patch = regular_automatic_extended_patch(monkeypatch)
    patch.set_radial_layers(3)
    patch.initialize_automatic_outer_geometry()
    straight_endpoints = (
        QPointF(patch.outer_nodes[0].position),
        QPointF(patch.outer_nodes[-1].position),
    )

    patch.enable_bezier_mode()

    assert patch.outer_nodes[0].position == straight_endpoints[0]
    assert patch.outer_nodes[-1].position == straight_endpoints[1]


def test_plain_b_keeps_multilayer_bezier_handles_and_samples_local(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    patch = regular_automatic_extended_patch(monkeypatch)
    for node in patch.ordered_nodes:
        node.xx = np.array([
            [node.position.x(), 1.0e6, -2.0e6, 0.0],
            [node.position.y(), -3.0e6, 4.0e6, 0.0],
        ])
    for element in grid_editor5.element_list:
        element.sizes = np.ones((4, 4))
        element.sizes[1, :] = 1.0e-7
    for edge in patch.ordered_edges:
        edge.sizes = np.array([[1.0, 1.0], [1.0e-7, 1.0e-7]])
    patch.set_radial_layers(4)
    patch.initialize_automatic_outer_geometry()
    straight_outer = [
        grid_editor5.np_point(node.position) for node in patch.outer_nodes
    ]
    scene.addItem(patch)
    view.current_extended_patch = patch

    view.keyPressEvent(SimpleNamespace(
        key=lambda: Qt.Key_B,
        modifiers=lambda: Qt.NoModifier,
    ))

    handles = {handle.role: handle for handle in patch.bezier_handles}
    start = grid_editor5.np_point(patch.bezier_start_position())
    end = grid_editor5.np_point(patch.bezier_end_position())
    chord = end - start
    start_delta = grid_editor5.np_point(
        handles["start_tangent"].pos()
    ) - start
    end_delta = grid_editor5.np_point(
        handles["end_tangent"].pos()
    ) - end
    assert np.linalg.norm(start_delta) == pytest.approx(
        np.linalg.norm(chord) / 3.0
    )
    assert np.linalg.norm(end_delta) == pytest.approx(
        np.linalg.norm(chord) / 3.0
    )
    assert np.allclose(start_delta, chord / 3.0)
    assert np.allclose(end_delta, -chord / 3.0)

    expected_inner_vectors = [
        grid_editor5.effective_node_basis_vector(
            node, patch.main_uv_index(),
            patch.ordered_edges[max(0, index - 1)],
        )
        for index, node in enumerate(patch.ordered_nodes)
    ]
    positional_inner_vectors = patch.positional_along_vectors(
        patch.ordered_nodes
    )
    expected_inner_vectors = [
        -vector if np.inner(vector, positional_inner_vectors[index]) < 0.0
        else vector
        for index, vector in enumerate(expected_inner_vectors)
    ]
    assert all(
        np.allclose(actual, expected)
        for actual, expected in zip(
            patch.preview_along_vectors[0], expected_inner_vectors
        )
    )
    expected_first_intermediate = [
        0.75 * inner + 0.25 * outer
        for inner, outer in zip(
            expected_inner_vectors, patch.preview_along_vectors[-1]
        )
    ]
    assert all(
        np.allclose(actual, expected)
        for actual, expected in zip(
            patch.preview_along_vectors[1], expected_first_intermediate
        )
    )
    assert max(
        np.linalg.norm(vector)
        for row in patch.preview_along_vectors
        for vector in row
    ) < 2.0 * np.linalg.norm(chord)

    envelope_points = np.array([
        grid_editor5.np_point(node.position) for node in patch.ordered_nodes
    ] + straight_outer)
    lower = np.min(envelope_points, axis=0) - 1.0e-9
    upper = np.max(envelope_points, axis=0) + 1.0e-9
    outer_samples = np.array([
        grid_editor5.np_point(node.position) for node in patch.outer_nodes
    ])
    assert np.all(outer_samples >= lower)
    assert np.all(outer_samples <= upper)
    assert max(
        np.linalg.norm(tangent) for tangent in patch.outer_tangents
    ) <= 2.0 * np.linalg.norm(chord)
    preview_bounds = patch.path().boundingRect()
    margin = np.linalg.norm(chord)
    assert preview_bounds.left() >= lower[0] - margin
    assert preview_bounds.right() <= upper[0] + margin
    assert preview_bounds.top() >= lower[1] - margin
    assert preview_bounds.bottom() <= upper[1] + margin
    assert app is not None


@pytest.mark.parametrize("radial_layers", [1, 2, 3])
def test_automatic_one_cap_scales_only_the_free_endpoint(
    monkeypatch, radial_layers
):
    patch = regular_automatic_extended_patch(monkeypatch)
    cap_nodes = [
        SimpleNamespace(
            index=20 + index,
            position=QPointF(0.0, 1.0 + index),
            xx=np.array([[0.0, 1.0 / 3.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0 / 3.0, 0.0]]),
        )
        for index in range(radial_layers + 1)
    ]
    fixed_outer_node = cap_nodes[-1]
    patch.one_cap_topology = SimpleNamespace(
        outer_start_node=fixed_outer_node,
        outer_end_node=None,
        start_cap_nodes=cap_nodes,
        end_cap_nodes=[],
        start_cap_edges=[object()] * radial_layers,
        end_cap_edges=[],
    )
    patch.radial_layers = radial_layers
    patch.initialize_automatic_outer_geometry()

    assert patch.outer_nodes[0] is fixed_outer_node
    assert patch.outer_nodes[-1].position == QPointF(
        2.0, 1.0 + radial_layers
    )
    assert patch.outer_nodes[1].position == QPointF(
        1.0, 1.0 + radial_layers
    )
    straight_free_position = QPointF(patch.outer_nodes[-1].position)

    patch.enable_bezier_mode()

    assert patch.outer_nodes[0] is fixed_outer_node
    assert patch.outer_nodes[-1].position == straight_free_position


def test_automatic_one_cap_regeneration_never_moves_fixed_endpoint(monkeypatch):
    patch = regular_automatic_extended_patch(monkeypatch)
    cap_nodes = [
        SimpleNamespace(
            index=20 + index,
            position=QPointF(0.0, 1.0 + index),
            xx=np.array([[0.0, 1.0 / 3.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0 / 3.0, 0.0]]),
        )
        for index in range(4)
    ]
    fixed_outer_node = cap_nodes[-1]
    patch.one_cap_topology = SimpleNamespace(
        outer_start_node=fixed_outer_node,
        outer_end_node=None,
        start_cap_nodes=cap_nodes,
        end_cap_nodes=[],
        start_cap_edges=[object()] * 3,
        end_cap_edges=[],
    )
    patch.radial_layers = 3
    patch.initialize_automatic_outer_geometry()

    patch.radial_layers = 2
    regenerated_start, regenerated_end = patch.automatic_outer_endpoints()

    assert regenerated_start == fixed_outer_node.position
    assert fixed_outer_node.position == QPointF(0.0, 4.0)
    assert regenerated_end == QPointF(2.0, 3.0)


def test_two_cap_automatic_initialization_leaves_outer_geometry_unchanged(
    monkeypatch
):
    patch = regular_automatic_extended_patch(monkeypatch)
    start_node = SimpleNamespace(index=20, position=QPointF(0.0, 3.0))
    end_node = SimpleNamespace(index=21, position=QPointF(2.0, 4.0))
    patch.capped_gap = SimpleNamespace(
        outer_start_node=start_node,
        outer_end_node=end_node,
        start_cap_nodes=[], end_cap_nodes=[],
        start_cap_edges=[object(), object()],
        end_cap_edges=[object(), object()],
    )
    patch.radial_layers = 2
    patch.set_outer_positions([
        start_node.position, QPointF(1.0, 3.5), end_node.position
    ])
    before = [QPointF(node.position) for node in patch.outer_nodes]

    patch.initialize_automatic_outer_geometry()

    assert [node.position for node in patch.outer_nodes] == before


def test_bezier_preview_along_vectors_interpolate_inner_to_outer():
    app = QApplication.instance() or QApplication([])
    inner_vectors = [
        np.array([0.25, 0.10]),
        np.array([0.30, 0.20]),
        np.array([0.35, 0.10]),
    ]
    inner_nodes = []
    for index, vector in enumerate(inner_vectors):
        xx = np.zeros((2, 4))
        xx[:, 0] = [float(index), 0.0]
        xx[:, 2] = vector
        inner_nodes.append(jorek_node_item(index, xx, 2))
    edges = [SimpleNamespace(uv_index=2) for unused_index in range(2)]
    patch = grid_editor5.extended_patch(inner_nodes, edges)
    patch.set_radial_layers(3)
    patch.set_outer_positions([
        QPointF(0.0, 3.0), QPointF(1.0, 3.4), QPointF(2.0, 3.0)
    ])
    patch.enable_bezier_mode()
    handles = {handle.role: handle for handle in patch.bezier_handles}
    handles["start_tangent"].setPos(QPointF(0.6, 4.2))
    handles["end_tangent"].setPos(QPointF(1.4, 4.2))
    patch.update_bezier_from_handles()

    scales = grid_editor5.bezier_nodal_parameter_scales(
        patch.outer_parameters
    )
    expected_outer = [
        scales[index] * np.asarray(tangent) / 3.0
        for index, tangent in enumerate(patch.outer_tangents)
    ]
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(
            patch.preview_along_vectors[0], inner_vectors
        )
    )
    assert all(
        np.allclose(actual, expected)
        for actual, expected in zip(
            patch.preview_along_vectors[-1], expected_outer
        )
    )
    for radial_index in (1, 2):
        fraction = radial_index / 3.0
        for column in range(3):
            assert np.allclose(
                patch.preview_along_vectors[radial_index][column],
                (1.0 - fraction) * inner_vectors[column]
                + fraction * expected_outer[column],
            )
    primitive_middle = patch.positional_along_vectors(
        patch.preview_node_rows[1]
    )[1]
    assert not np.allclose(
        patch.preview_along_vectors[1][1], primitive_middle
    )
    assert app is not None


def test_manual_radial_preview_builds_incrementally_per_outer_column():
    nodes_xx = np.zeros((2, 4, 3))
    nodes_xx[:, 0, :] = [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 1)
        for index in range(3)
    ]
    edges = [
        SimpleNamespace(
            nodes=[nodes[index], nodes[index + 1]],
            vertices=[index, index + 1], uv_index=1,
            element_index=100 + index,
        )
        for index in range(2)
    ]
    patch = grid_editor5.extended_patch(nodes, edges)
    patch.set_radial_layers(4)

    patch.add_outer_node(QPointF(0.0, 4.0))

    assert len(patch.outer_nodes) == 1
    assert len(patch.preview_node_rows) == 5
    assert all(len(row) == 1 for row in patch.preview_node_rows)
    assert [
        row[0].position.y() for row in patch.preview_node_rows
    ] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    first_column_path_elements = patch.path().elementCount()
    assert first_column_path_elements > 0

    patch.add_outer_node(QPointF(1.0, 5.0))

    assert len(patch.outer_nodes) == 2
    assert all(len(row) == 2 for row in patch.preview_node_rows)
    assert patch.path().elementCount() > first_column_path_elements

    patch.add_outer_node(QPointF(2.0, 4.0))

    assert len(patch.outer_nodes) == patch.required_outer_node_count
    assert all(len(row) == 3 for row in patch.preview_node_rows)


def test_basis_drag_updates_elements_and_only_matching_boundary_edges():
    app = QApplication.instance() or QApplication([])
    node = jorek_node_item(0, np.zeros((2, 4)), 0)
    element = UpdateRecorder()
    blue_edge = UpdateRecorder(uv_index=1)
    red_edge = UpdateRecorder(uv_index=2)
    node.connected_elements = [element]
    node.connected_boundary_edges = [blue_edge, red_edge]

    node.update_connected_items(basis_index=1)

    assert element.update_count == 1
    assert blue_edge.update_count == 1
    assert red_edge.update_count == 0
    assert app is not None


def test_node_bounds_contain_vector_endpoints_and_handle_margin():
    app = QApplication.instance() or QApplication([])
    xx = np.zeros((2, 4))
    xx[:, 0] = [100.0, 200.0]
    xx[:, 1] = [80.0, -40.0]
    xx[:, 2] = [-60.0, 90.0]
    node = jorek_node_item(0, xx, 0)

    bounds = node.boundingRect()

    assert bounds.contains(QPointF(100.0, 200.0))
    assert bounds.contains(QPointF(180.0 + 8.0, 160.0))
    assert bounds.contains(QPointF(40.0 - 8.0, 290.0))
    assert app is not None


@pytest.mark.parametrize("zoom_level", [0.1, 1.0, 10.0])
def test_node_shape_only_contains_visible_marker_at_any_zoom(zoom_level):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    view.zoom_level = zoom_level
    xx = np.zeros((2, 4))
    xx[:, 0] = [100.0, 200.0]
    xx[:, 1] = [1000.0, 0.0]
    xx[:, 2] = [0.0, 1000.0]
    node = jorek_node_item(0, xx, 0)
    scene.addItem(node)

    radius = grid_editor5.NODE_MARKER_SIZE / (2.0 * zoom_level)
    inside = node.position + QPointF(0.99 * radius, 0.0)
    outside = node.position + QPointF(1.01 * radius, 0.0)
    blue_vector_middle = node.position + QPointF(500.0, 0.0)

    assert node.shape().contains(inside)
    assert not node.shape().contains(outside)
    assert not node.shape().contains(blue_vector_middle)
    assert node in scene.items(inside)
    assert node not in scene.items(outside)
    assert node not in scene.items(blue_vector_middle)
    assert node.blue_handle in scene.items(node.blue_handle.scenePos())
    assert node.red_handle in scene.items(node.red_handle.scenePos())
    assert node.blue_handle.pen().isCosmetic()
    assert node.red_handle.pen().isCosmetic()
    assert node.blue_handle.pen().widthF() == pytest.approx(
        grid_editor5.GRAPHICS_HANDLE_OUTLINE_WIDTH
    )
    assert node.red_handle.pen().widthF() == pytest.approx(
        grid_editor5.GRAPHICS_HANDLE_OUTLINE_WIDTH
    )
    assert node.boundingRect().contains(node.blue_handle.pos())
    assert node.boundingRect().contains(node.red_handle.pos())
    assert app is not None


def test_basis_vector_handles_are_fixed_screen_items_at_construction():
    app = QApplication.instance() or QApplication([])
    node = jorek_node_item(0, np.zeros((2, 4)), 0)

    for handle in (node.blue_handle, node.red_handle):
        assert (
            handle.flags()
            & grid_editor5.QGraphicsItem.ItemIgnoresTransformations
        )
        assert handle.rect().width() == pytest.approx(
            grid_editor5.VECTOR_HANDLE_SIZE
        )
        assert handle.rect().height() == pytest.approx(
            grid_editor5.VECTOR_HANDLE_SIZE
        )
    assert app is not None


def test_node_repaint_and_zoom_do_not_mutate_basis_handle_geometry():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.resize(320, 240)
    view.setScene(scene)
    xx = np.zeros((2, 4))
    xx[:, 1] = [2.0, 0.0]
    xx[:, 2] = [0.0, 2.0]
    node = jorek_node_item(0, xx, 0)
    scene.addItem(node)
    view.show()
    original_rects = (
        node.blue_handle.rect(), node.red_handle.rect()
    )

    for zoom_level in (0.1, 1.0, 10.0):
        view.zoom_level = zoom_level
        view.setTransform(QTransform().scale(zoom_level, zoom_level))
        node.update()
        view.viewport().repaint()
        app.processEvents()

        assert node.blue_handle.rect() == original_rects[0]
        assert node.red_handle.rect() == original_rects[1]
        for handle in (node.blue_handle, node.red_handle):
            screen_rect = handle.deviceTransform(
                view.viewportTransform()
            ).mapRect(handle.rect())
            assert screen_rect.width() == pytest.approx(
                grid_editor5.VECTOR_HANDLE_SIZE
            )
            assert screen_rect.height() == pytest.approx(
                grid_editor5.VECTOR_HANDLE_SIZE
            )
            assert handle in view.items(
                view.mapFromScene(handle.scenePos())
            )
    assert app is not None


@pytest.mark.parametrize(
    "depth, expected",
    [
        (0, {
            index for index in range(25)
            if index // 5 in (0, 4) or index % 5 in (0, 4)
        }),
        (1, {
            index for index in range(25)
            if index // 5 in (0, 1, 3, 4) or index % 5 in (0, 1, 3, 4)
        }),
        (2, set(range(25))),
    ],
)
def test_editable_depth_expands_complete_edge_rings(depth, expected):
    elements, boundary_edges = structured_editable_topology()

    editable = grid_editor5.editable_element_indices(
        elements, boundary_edges, depth=depth
    )

    assert editable == expected
    if depth == 0:
        assert editable == grid_editor5.editable_boundary_element_indices(
            elements, boundary_edges
        )
    assert grid_editor5.editable_node_indices(elements, editable) == {
        vertex
        for element in elements if element.index in expected
        for vertex in element.vertices
    }


def test_editable_adjacency_excludes_node_only_and_inactive_contacts():
    elements = [
        SimpleNamespace(index=0, vertices=[0, 1, 2, 3], active=True),
        SimpleNamespace(index=1, vertices=[2, 4, 5, 6], active=True),
        SimpleNamespace(index=2, vertices=[1, 7, 8, 2], active=False),
        SimpleNamespace(index=3, vertices=[7, 9, 10, 8], active=True),
    ]
    boundary_edges = [SimpleNamespace(element_index=0, active=True)]

    adjacency = grid_editor5.element_edge_adjacency(elements)
    editable = grid_editor5.editable_element_indices(
        elements, boundary_edges, depth=9, adjacency=adjacency
    )

    assert adjacency[0] == set()
    assert 2 not in adjacency
    assert editable == {0}


def test_one_edge_preview_stores_and_draws_the_endpoint_pairing():
    app = QApplication.instance() or QApplication([])
    old_node0 = type("Node", (), {"index": 4, "position": QPointF(0.0, 0.0)})()
    old_node1 = type("Node", (), {"index": 5, "position": QPointF(10.0, 0.0)})()
    edge = type(
        "Edge",
        (),
        {
            "nodes": [old_node0, old_node1],
            "vertices": [4, 5],
            "element_index": 2,
            "element_side": 1,
            "path": lambda self: QPainterPath(),
        },
    )()
    grid_editor5.view = type("View", (), {"selected_edges": [edge]})()
    patch = big_patch()

    patch.add_corner(big_patch_node(QPointF(9.0, 5.0)))
    patch.add_corner(big_patch_node(QPointF(1.0, 5.0)))

    assert patch.corner_nodes[0].position == QPointF(1.0, 5.0)
    assert patch.corner_nodes[1].position == QPointF(9.0, 5.0)
    assert patch.path().currentPosition() == old_node0.position
    assert app is not None


def test_two_edge_preview_draws_outer_shared_outer_new_cycle():
    app = QApplication.instance() or QApplication([])
    shared = type("Node", (), {"index": 10, "position": QPointF(0.0, 0.0)})()
    outer0 = type("Node", (), {"index": 11, "position": QPointF(-5.0, 0.0)})()
    outer1 = type("Node", (), {"index": 12, "position": QPointF(0.0, 5.0)})()
    edge0 = type(
        "Edge", (), {"nodes": [outer0, shared], "path": lambda self: QPainterPath()}
    )()
    edge1 = type(
        "Edge", (), {"nodes": [shared, outer1], "path": lambda self: QPainterPath()}
    )()
    grid_editor5.view = type("View", (), {"selected_edges": [edge0, edge1]})()
    patch = big_patch()

    patch.add_corner(big_patch_node(QPointF(-5.0, 5.0)))

    path = patch.path()
    points = [
        QPointF(path.elementAt(i).x, path.elementAt(i).y)
        for i in range(path.elementCount())
    ]
    assert points[-5:] == [
        outer0.position,
        shared.position,
        outer1.position,
        QPointF(-5.0, 5.0),
        outer0.position,
    ]
    assert app is not None


def test_three_edge_preview_closes_the_topologically_ordered_chain():
    app = QApplication.instance() or QApplication([])
    nodes = [
        type("Node", (), {"index": i, "position": QPointF(float(i), float(i % 2))})()
        for i in range(1, 5)
    ]

    def edge(node0, node1, uv_index):
        return type(
            "Edge",
            (),
            {
                "nodes": [node0, node1],
                "uv_index": uv_index,
                "path": lambda self: QPainterPath(),
            },
        )()

    selected_edges = [
        edge(nodes[1], nodes[2], 2),
        edge(nodes[2], nodes[3], 1),
        edge(nodes[0], nodes[1], 1),
    ]
    grid_editor5.view = type("View", (), {"selected_edges": selected_edges})()

    patch = big_patch()
    path = patch.path()
    points = [
        QPointF(path.elementAt(i).x, path.elementAt(i).y)
        for i in range(path.elementCount())
    ]

    assert points[-5:] == [node.position for node in nodes] + [nodes[0].position]
    assert patch.corner_nodes == []
    assert app is not None


def configure_element_creation(monkeypatch, positions, edge_specs):
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, len(positions)))
    for index, position in enumerate(positions):
        nodes_xx[:, 0, index] = position
        nodes_xx[:, 1, index] = [1.0, 0.0]
        nodes_xx[:, 2, index] = [0.0, 1.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 2)
        for index in range(len(positions))
    ]

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)

    old_edges = []
    for node0, node1, uv_index, element_index in edge_specs:
        edge = grid_editor5.boundary_edge(
            [nodes[node0], nodes[node1]], [node0, node1], [0, 1],
            element_index, 0, uv_index, np.ones((2, 2)),
        )
        old_edges.append(edge)
        scene.addItem(edge)
    install_edge_owner_lookup(monkeypatch, old_edges)

    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(
        grid_editor5, "boundary_list", list(old_edges), raising=False
    )
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)
    view.selected_edges = list(old_edges)
    patch = big_patch()
    scene.addItem(patch)
    view.current_patch = patch
    return scene, view, nodes, old_edges, patch


def assert_boundary_status_matches_edges(nodes, edges):
    for node in nodes:
        directions = {
            edge.uv_index for edge in edges if node.index in edge.vertices
        }
        if not directions:
            assert node.boundary == 0
        elif len(directions) == 1:
            assert node.boundary == next(iter(directions))
        else:
            assert node.boundary == 2


def configure_element_deletion(
    monkeypatch, positions, element_vertices, boundary_sides
):
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes_xx = np.zeros((2, 4, len(positions)))
    for index, position in enumerate(positions):
        nodes_xx[:, 0, index] = position
        nodes_xx[:, 1, index] = [1.0, 0.0]
        nodes_xx[:, 2, index] = [0.0, 1.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 2)
        for index in range(len(positions))
    ]
    for node in nodes:
        scene.addItem(node)

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)

    elements = [
        grid_editor5.jorek_element_item(index, vertices, np.ones((4, 4)))
        for index, vertices in enumerate(element_vertices)
    ]
    for element in elements:
        scene.addItem(element)
    monkeypatch.setattr(grid_editor5, "element_list", elements, raising=False)

    edges = []
    for element_index, side in boundary_sides:
        edge = grid_editor5.boundary_edge_from_element_side(
            elements[element_index], side
        )
        edges.append(edge)
        elements[element_index].edges.append(edge)
        scene.addItem(edge)
    monkeypatch.setattr(grid_editor5, "boundary_list", edges, raising=False)
    grid_editor5.recompute_node_boundaries(nodes, edges)
    grid_editor5.rebuild_node_connections()
    return scene, view, nodes, elements, edges


def assert_no_dangling_grid_references():
    element_indices = {element.index for element in grid_editor5.element_list}
    referenced_nodes = {
        vertex
        for element in grid_editor5.element_list
        for vertex in element.vertices
    }
    for element in grid_editor5.element_list:
        assert all(0 <= vertex < len(grid_editor5.node_list) for vertex in element.vertices)
    for edge in grid_editor5.boundary_list:
        assert set(edge.vertices).issubset(referenced_nodes)
        assert edge.element_index in element_indices
        assert all(node is grid_editor5.node_list[node.index] for node in edge.nodes)
    for node in grid_editor5.node_list:
        assert all(
            element in grid_editor5.element_list
            for element in node.connected_elements
        )
        assert all(
            edge in grid_editor5.boundary_list
            for edge in node.connected_boundary_edges
        )


def test_delete_outer_boundary_element(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene, view, nodes, elements, old_edges = configure_element_deletion(
        monkeypatch,
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        [[0, 1, 2, 3]],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    )
    deleted_element = elements[0]
    view.selected_elements = [deleted_element]

    grid_editor5.delete_selected_element(view)

    assert grid_editor5.element_list == []
    assert grid_editor5.boundary_list == []
    assert deleted_element.scene() is scene
    assert not deleted_element.isVisible()
    assert not deleted_element.path_item.isVisible()
    assert all(edge.scene() is scene for edge in old_edges)
    assert all(not edge.isVisible() for edge in old_edges)
    assert all(node.boundary == 0 for node in nodes)
    assert all(node.scene() is scene for node in nodes)
    assert all(not node.isVisible() for node in nodes)
    assert all(not node.isEnabled() for node in nodes)
    assert all(node.blue_handle in scene.items() for node in nodes)
    assert all(node.red_handle in scene.items() for node in nodes)
    assert_no_dangling_grid_references()
    assert app is not None


@pytest.mark.parametrize("selection", [[], [object(), object()]])
def test_delete_requires_exactly_one_selected_element(selection, capsys):
    view = SimpleNamespace(selected_elements=selection)

    grid_editor5.delete_selected_element(view)

    assert "Select exactly one element to delete" in capsys.readouterr().out


def test_delete_element_exposes_internal_side(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene, view, nodes, elements, old_edges = configure_element_deletion(
        monkeypatch,
        [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        ],
        [[0, 1, 4, 3], [1, 2, 5, 4]],
        [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
    )
    deleted_element, remaining_element = elements
    view.selected_elements = [deleted_element]
    view.editable_depth = 2

    grid_editor5.delete_selected_element(view)

    assert grid_editor5.element_list == [remaining_element]
    assert deleted_element.scene() is scene
    assert not deleted_element.isVisible()
    assert not deleted_element.path_item.isVisible()
    assert len(grid_editor5.boundary_list) == 4
    exposed_edges = [
        edge for edge in grid_editor5.boundary_list
        if frozenset(edge.vertices) == frozenset((1, 4))
    ]
    assert len(exposed_edges) == 1
    assert exposed_edges[0].element_index == remaining_element.index
    assert exposed_edges[0].element_side == 3
    assert exposed_edges[0].scene() is scene
    assert view.editable_depth == 2
    assert view.editable_element_indices_set == {remaining_element.index}
    assert nodes[0].boundary == 0
    assert nodes[3].boundary == 0
    assert nodes[0].scene() is scene
    assert nodes[3].scene() is scene
    assert not nodes[0].isVisible()
    assert not nodes[3].isVisible()
    assert not nodes[0].isEnabled()
    assert not nodes[3].isEnabled()
    assert all(nodes[index].scene() is scene for index in (1, 2, 4, 5))
    assert_boundary_status_matches_edges(nodes, grid_editor5.boundary_list)
    assert_no_dangling_grid_references()
    assert app is not None


def test_delete_neighbor_after_orphan_node_graphics_deactivated(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene, view, nodes, elements, unused_edges = configure_element_deletion(
        monkeypatch,
        [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        ],
        [[0, 1, 4, 3], [1, 2, 5, 4]],
        [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
    )
    first_element, neighboring_element = elements
    view.selected_nodes = list(nodes)
    view.selected_elements = [first_element]

    grid_editor5.delete_selected_element(view)
    app.processEvents()

    inactive_vector = grid_editor5.jorek.nodes_xx[:, 1, 0].copy()
    nodes[0].blue_handle.move_to_scene(QPointF(99.0, 99.0))
    assert np.array_equal(
        grid_editor5.jorek.nodes_xx[:, 1, 0], inactive_vector
    )
    assert nodes[0] not in view.selected_nodes

    view.selected_elements = [neighboring_element]
    grid_editor5.delete_selected_element(view)
    app.processEvents()

    scene_items = scene.items()
    assert grid_editor5.element_list == []
    assert grid_editor5.boundary_list == []
    assert view.selected_nodes == []
    assert all(element.scene() is scene for element in elements)
    assert all(not element.active and not element.isVisible() for element in elements)
    assert all(not element.path_item.isVisible() for element in elements)
    assert all(edge.scene() is scene for edge in unused_edges)
    assert all(not edge.active and not edge.isVisible() for edge in unused_edges)
    for node in nodes:
        assert node.scene() is scene
        assert not node.isVisible()
        assert not node.isEnabled()
        assert not node.active
        assert node.index == nodes.index(node)
        assert node.connected_elements == []
        assert node.connected_boundary_edges == []
        assert node.blue_handle in scene_items
        assert node.red_handle in scene_items
        assert not node.blue_handle.isVisible()
        assert not node.red_handle.isVisible()


def test_delete_several_adjacent_elements_keeps_inactive_graphics(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene, view, nodes, elements, original_edges = configure_element_deletion(
        monkeypatch,
        [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0),
        ],
        [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]],
        [
            (0, 0), (0, 2), (0, 3),
            (1, 0), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ],
    )
    obsolete_edges = []

    all_elements = list(elements)
    for deleted_element in all_elements:
        previous_edges = list(grid_editor5.boundary_list)
        view.selected_elements = [deleted_element]
        grid_editor5.delete_selected_element(view)
        app.processEvents()
        obsolete_edges.extend(
            edge for edge in previous_edges
            if edge not in grid_editor5.boundary_list
        )
        assert all(element.active for element in grid_editor5.element_list)
        assert all(edge.active for edge in grid_editor5.boundary_list)
        assert_no_dangling_grid_references()

    assert grid_editor5.element_list == []
    assert grid_editor5.boundary_list == []
    assert all(element.scene() is scene for element in all_elements)
    assert all(
        not element.active and not element.isVisible()
        for element in all_elements
    )
    assert all(edge.scene() is scene for edge in obsolete_edges)
    assert all(not edge.active and not edge.isVisible() for edge in obsolete_edges)
    assert all(node.scene() is scene and not node.active for node in nodes)


def test_one_edge_creation_recomputes_boundary_state(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene, unused_view, nodes, old_edges, patch = configure_element_creation(
        monkeypatch, [(0.0, 0.0), (1.0, 0.0)], [(0, 1, 1, 10)]
    )
    patch.add_corner(big_patch_node(QPointF(1.0, 1.0)))
    patch.add_corner(big_patch_node(QPointF(0.0, 1.0)))

    grid_editor5.add_patch_to_nodes_elements(patch)

    assert len(grid_editor5.boundary_list) == 3
    assert old_edges[0] not in grid_editor5.boundary_list
    assert old_edges[0].scene() is scene
    assert not old_edges[0].active
    assert not old_edges[0].isVisible()
    assert all(edge.scene() is scene for edge in grid_editor5.boundary_list)
    assert_boundary_status_matches_edges(
        grid_editor5.node_list, grid_editor5.boundary_list
    )
    assert len(grid_editor5.node_list) == 4
    assert app is not None


def test_two_edge_creation_recomputes_boundary_state(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene, unused_view, nodes, old_edges, patch = configure_element_creation(
        monkeypatch,
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        [(0, 1, 1, 10), (1, 2, 2, 11)],
    )
    patch.add_corner(big_patch_node(QPointF(0.0, 1.0)))

    grid_editor5.add_patch_to_nodes_elements(patch)

    assert len(grid_editor5.boundary_list) == 2
    assert all(edge not in grid_editor5.boundary_list for edge in old_edges)
    assert all(edge.scene() is scene for edge in old_edges)
    assert all(not edge.active and not edge.isVisible() for edge in old_edges)
    assert all(edge.scene() is scene for edge in grid_editor5.boundary_list)
    assert nodes[1].boundary == 0
    assert_boundary_status_matches_edges(
        grid_editor5.node_list, grid_editor5.boundary_list
    )
    assert len(grid_editor5.node_list) == 4
    assert app is not None


def test_three_edge_patch_creates_element_and_updates_boundary(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)

    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    nodes_xx = np.zeros((2, 4, 4))
    for index, position in enumerate(positions):
        nodes_xx[:, 0, index] = position
        nodes_xx[:, 1, index] = [1.0, 0.0]
        nodes_xx[:, 2, index] = [0.0, 1.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 1)
        for index in range(4)
    ]

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)

    def edge(node0, node1, uv_index, element_index):
        sizes = np.ones((2, 2))
        return grid_editor5.boundary_edge(
            [node0, node1], [node0.index, node1.index],
            [0, 1], element_index, 0, uv_index, sizes,
        )

    old_edges = [
        edge(nodes[0], nodes[1], 1, 10),
        edge(nodes[1], nodes[2], 2, 11),
        edge(nodes[2], nodes[3], 1, 12),
    ]
    for old_edge in old_edges:
        scene.addItem(old_edge)

    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(
        grid_editor5, "boundary_list", list(old_edges), raising=False
    )
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)

    view.selected_edges = list(old_edges)
    patch = big_patch()
    scene.addItem(patch)
    view.current_patch = patch

    grid_editor5.add_patch_to_nodes_elements(patch)

    assert len(grid_editor5.element_list) == 1
    assert grid_editor5.element_list[0].vertices == [0, 1, 2, 3]
    assert all(old_edge not in grid_editor5.boundary_list for old_edge in old_edges)
    assert all(old_edge.scene() is scene for old_edge in old_edges)
    assert all(not old_edge.active and not old_edge.isVisible() for old_edge in old_edges)
    assert len(grid_editor5.boundary_list) == 1
    missing_edge = grid_editor5.boundary_list[0]
    assert missing_edge.vertices == [3, 0]
    assert missing_edge.uv_index == old_edges[1].uv_index
    assert missing_edge.scene() is scene
    assert nodes[0].boundary != 0
    assert nodes[1].boundary == 0
    assert nodes[2].boundary == 0
    assert nodes[3].boundary != 0
    assert_boundary_status_matches_edges(nodes, grid_editor5.boundary_list)
    assert len(grid_editor5.node_list) == 4
    assert app is not None


def test_escape_clears_all_selections_and_current_patch():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    edge = QGraphicsPathItem()
    edge.setPen(QPen(Qt.green, 2.0))
    patch = QGraphicsPathItem()
    scene.addItem(edge)
    scene.addItem(patch)
    view.selected_edges = [edge]
    boundary_node_item = QGraphicsEllipseItem()
    boundary_node_item.setBrush(QBrush(QColor(0, 255, 0)))
    boundary_node = SimpleNamespace(
        boundary=2, ellipse_item=boundary_node_item, update=lambda: None
    )
    interior_node_item = QGraphicsEllipseItem()
    interior_node_item.setBrush(QBrush(QColor(0, 255, 0)))
    interior_node = SimpleNamespace(
        boundary=0, ellipse_item=interior_node_item, update=lambda: None
    )
    element_path = QGraphicsPathItem()
    element_path.setBrush(QBrush(QColor(50, 50, 50, 64)))
    element = SimpleNamespace(path_item=element_path, update=lambda: None)
    view.selected_nodes = [boundary_node, interior_node]
    view.selected_elements = [element]
    view.current_patch = patch
    pending_event = type("KeyEvent", (), {"key": lambda self: Qt.Key_1})()
    view.keyPressEvent(pending_event)
    assert view.pending_main_uv_index == 1
    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_Escape})()

    view.keyPressEvent(event)

    assert view.selected_edges == []
    assert edge.pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert edge.pen().widthF() == pytest.approx(
        grid_editor5.BOUNDARY_EDGE_WIDTH
    )
    assert edge.pen().isCosmetic()
    assert view.current_patch is None
    assert view.pending_main_uv_index is None
    assert patch.scene() is None
    assert view.selected_nodes == []
    assert view.selected_elements == []
    assert boundary_node_item.brush().color() == grid_editor5.NODE_COLOR
    assert interior_node_item.brush().color() == grid_editor5.NODE_COLOR
    assert element_path.brush().color() == QColor(255, 255, 255, 64)
    assert app is not None


def ambiguous_two_plus_two_selection(scene):
    nodes = {
        index: SimpleNamespace(index=index, position=position)
        for index, position in {
            0: QPointF(0.0, 0.0),
            1: QPointF(1.0, 0.0),
            2: QPointF(2.0, 0.0),
            3: QPointF(0.0, 1.0),
            4: QPointF(0.0, 2.0),
        }.items()
    }

    def edge(start, end, uv_index):
        item = QGraphicsPathItem()
        item.nodes = [nodes[start], nodes[end]]
        item.vertices = [start, end]
        item.uv_index = uv_index
        item.element_index = start
        item.element_side = 0
        scene.addItem(item)
        return item

    edges_by_uv = {
        1: [edge(0, 1, 1), edge(1, 2, 1)],
        2: [edge(0, 3, 2), edge(3, 4, 2)],
    }
    selected_edges = edges_by_uv[2][::-1] + edges_by_uv[1][::-1]
    return selected_edges, edges_by_uv


@pytest.mark.parametrize("main_uv_index", [1, 2])
def test_pending_direction_resolves_ambiguous_extended_patch(main_uv_index):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    view.set_pending_bezier_mode(False)
    selected_edges, edges_by_uv = ambiguous_two_plus_two_selection(scene)
    view.selected_edges = selected_edges

    direction_event = type(
        "KeyEvent", (),
        {"key": lambda self: Qt.Key_0 + main_uv_index},
    )()
    view.keyPressEvent(direction_event)
    assert view.pending_main_uv_index == main_uv_index

    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_E})())

    assert view.current_extended_patch is not None
    assert view.current_extended_patch.ordered_edges == edges_by_uv[main_uv_index]
    assert view.pending_main_uv_index is None
    assert app is not None


def test_ambiguous_extended_patch_requests_explicit_direction(capsys):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    view.selected_edges, unused_edges_by_uv = ambiguous_two_plus_two_selection(
        scene
    )

    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_E})())

    output = capsys.readouterr().out
    assert view.current_extended_patch is None
    assert "Ambiguous extended patch" in output
    assert "Press 1 or 2" in output
    assert view.pending_main_uv_index is None
    assert app is not None


def test_e_starts_same_direction_extended_patch_and_escape_cancels(
    monkeypatch, capsys
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    view.set_pending_bezier_mode(False)
    nodes = {
        index: SimpleNamespace(
            index=index, position=QPointF(float(index - 178), 0.0)
        )
        for index in (178, 179, 180)
    }

    def edge(start, end):
        item = QGraphicsPathItem()
        item.nodes = [nodes[start], nodes[end]]
        item.vertices = [start, end]
        item.uv_index = 1
        item.element_index = start
        item.setPen(QPen(Qt.green, 2.0))
        scene.addItem(item)
        return item

    first_edge = edge(179, 178)
    second_edge = edge(180, 179)
    view.selected_edges = [second_edge, first_edge]
    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_E})()

    view.keyPressEvent(event)

    state = view.current_extended_patch
    assert state is not None
    assert [node.index for node in state.ordered_nodes] == [178, 179, 180]
    assert state.ordered_edges == [first_edge, second_edge]
    output = capsys.readouterr().out
    assert "[178, 179, 180]" in output
    assert "[179, 178]" in output
    assert "[180, 179]" in output

    monkeypatch.setattr(
        grid_editor5.QApplication,
        "keyboardModifiers",
        lambda: Qt.ControlModifier,
    )
    click_positions = [QPoint(319, 249), QPoint(320, 249), QPoint(321, 249)]
    expected_scene_positions = [view.mapToScene(point) for point in click_positions]
    for position in click_positions:
        mouse_event = SimpleNamespace(pos=lambda position=position: position)
        view.mousePressEvent(mouse_event)
    assert view.current_extended_patch is state
    assert view.current_patch is None
    assert len(state.outer_nodes) == 3
    assert [node.position for node in state.outer_nodes] == expected_scene_positions
    assert state.path().elementCount() == 12

    extra_click = SimpleNamespace(pos=lambda: QPoint(30, 10))
    view.mousePressEvent(extra_click)
    assert len(state.outer_nodes) == 3

    escape_event = type(
        "KeyEvent", (), {"key": lambda self: Qt.Key_Escape}
    )()
    view.keyPressEvent(escape_event)
    assert view.current_extended_patch is None
    assert state.scene() is None
    assert view.selected_edges == []
    assert first_edge.pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert second_edge.pen().color() == grid_editor5.BOUNDARY_EDGE_COLOR
    assert app is not None


def test_e_previews_capped_boundary_gap():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    view.set_pending_bezier_mode(False)
    nodes = {
        index: SimpleNamespace(
            index=index, position=position,
            xx=np.array([[0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0]]),
        )
        for index, position in {
            0: QPointF(0.0, 2.0), 1: QPointF(0.0, 0.0),
            2: QPointF(1.0, 0.0), 3: QPointF(2.0, 0.0),
            4: QPointF(2.0, 2.0),
        }.items()
    }

    def edge(start, end, uv_index):
        item = QGraphicsPathItem()
        item.nodes = [nodes[start], nodes[end]]
        item.vertices = [start, end]
        item.uv_index = uv_index
        item.element_index = start
        item.element_side = 0
        scene.addItem(item)
        return item

    main0 = edge(1, 2, 1)
    main1 = edge(2, 3, 1)
    start_side = edge(0, 1, 2)
    end_side = edge(3, 4, 2)
    view.selected_edges = [end_side, main1, start_side, main0]

    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_E})()
    view.keyPressEvent(event)

    patch = view.current_extended_patch
    assert patch is not None
    assert patch.can_commit is True
    assert patch.capped_gap.start_side_edge is start_side
    assert patch.capped_gap.end_side_edge is end_side
    assert [node.position for node in patch.outer_nodes] == [
        QPointF(0.0, 2.0), QPointF(1.0, 2.0), QPointF(2.0, 2.0)
    ]
    assert patch.path().elementCount() == 12

    assert view.current_extended_patch is patch
    assert app is not None


def test_one_cap_straight_e_waits_for_manual_free_endpoint(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    view.set_pending_bezier_mode(False)
    controls = grid_editor5.extended_patch_controls(view)
    nodes = {
        index: SimpleNamespace(
            index=index, position=position,
            xx=np.array([[0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0]]),
        )
        for index, position in {
            0: QPointF(0.0, 2.0), 1: QPointF(0.0, 0.0),
            2: QPointF(1.0, 0.0), 3: QPointF(2.0, 0.0),
        }.items()
    }

    def edge(start, end, uv_index):
        item = QGraphicsPathItem()
        item.nodes = [nodes[start], nodes[end]]
        item.vertices = [start, end]
        item.uv_index = uv_index
        item.element_index = start
        item.element_side = 0
        scene.addItem(item)
        return item

    view.selected_edges = [
        edge(2, 3, 1), edge(0, 1, 2), edge(2, 1, 1)
    ]
    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_E})()
    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_1})())

    view.keyPressEvent(event)

    patch = view.current_extended_patch
    assert patch.can_commit is True
    assert patch.bezier_mode is False
    assert patch.one_cap_topology.outer_start_node is nodes[0]
    assert patch.outer_nodes == []
    assert not patch.automatic_outer_geometry
    assert patch.add_outer_node(QPointF(2.0, 2.0))
    assert [node.position for node in patch.outer_nodes] == [
        QPointF(0.0, 2.0), QPointF(1.0, 2.0), QPointF(2.0, 2.0)
    ]
    assert not patch.automatic_outer_geometry
    assert not patch.add_outer_node(QPointF(3.0, 2.0))

    controls.update_from_view()
    assert view.enable_current_patch_bezier()

    roles = {handle.role for handle in patch.bezier_handles}
    assert patch.outer_nodes[0] is nodes[0]
    assert roles == {"end", "end_tangent", "cap_global_tangent"}
    assert patch.bezier_start_position() == nodes[0].position
    assert np.allclose(patch.bezier_start_vector(), nodes[0].xx[:, 1])
    assert controls.outer_boundary_combo.currentData() is True
    assert not controls.outer_boundary_combo.isEnabled()

    assert app is not None


def test_mixed_uv_topology_failure_does_not_fall_back_to_zero_cap(capsys):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(3)
    ]

    def edge(start, end, uv_index):
        return SimpleNamespace(
            nodes=[nodes[start], nodes[end]], vertices=[start, end],
            uv_index=uv_index, element_index=start, element_side=0,
        )

    view.selected_edges = [edge(0, 1, 1), edge(1, 2, 2)]
    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_E})()

    view.keyPressEvent(event)

    output = capsys.readouterr().out
    assert view.current_extended_patch is None
    assert "extended topology detection failed:" in output
    assert "ambiguous" in output
    assert "patch creation aborted" in output
    assert "Press 1 or 2" in output
    assert app is not None


@pytest.mark.parametrize("cap_edge_count", [1, 2, 4])
def test_multi_edge_one_cap_topology_and_preview_reuse_existing_nodes(
    cap_edge_count, monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    nodes = {}
    positions = {
        0: QPointF(0.0, 0.0),
        1: QPointF(1.0, 0.0),
        2: QPointF(2.0, 0.0),
    }
    for radial_index in range(1, cap_edge_count + 1):
        positions[2 + radial_index] = QPointF(0.0, float(radial_index))
    for index, position in positions.items():
        nodes[index] = SimpleNamespace(
            index=index, position=position,
            xx=np.array([[0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0]]),
        )

    def edge(start, end, uv_index):
        item = QGraphicsPathItem()
        item.nodes = [nodes[start], nodes[end]]
        item.vertices = [start, end]
        item.uv_index = uv_index
        item.element_index = start
        item.element_side = 0
        scene.addItem(item)
        return item

    main_edges = [edge(0, 1, 1), edge(1, 2, 1)]
    cap_edges = []
    previous = 0
    for radial_index in range(1, cap_edge_count + 1):
        current = 2 + radial_index
        cap_edges.append(edge(previous, current, 2))
        previous = current
    selected = list(reversed(cap_edges)) + list(reversed(main_edges))
    view.selected_edges = selected
    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_1})())
    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_E})())

    patch = view.current_extended_patch
    topology = patch.one_cap_topology
    assert patch.radial_layers == cap_edge_count
    patch.set_radial_layers(9 if cap_edge_count != 9 else 8)
    assert patch.radial_layers == cap_edge_count
    assert topology.start_cap_edges == cap_edges
    assert topology.end_cap_edges == []
    expected_cap_nodes = [nodes[0]] + [nodes[3 + i] for i in range(cap_edge_count)]
    assert topology.start_cap_nodes == expected_cap_nodes

    assert patch.add_outer_node(QPointF(2.0, float(cap_edge_count)))
    assert len(patch.preview_node_rows) == cap_edge_count + 1
    for row_index, cap_node in enumerate(expected_cap_nodes):
        assert patch.preview_node_rows[row_index][0] is cap_node

    patch.enable_bezier_mode()
    assert {handle.role for handle in patch.bezier_handles} == {
        "end", "end_tangent", "cap_global_tangent"
    }
    assert patch.outer_nodes[0] is expected_cap_nodes[-1]
    assert app is not None


def test_one_cap_bezier_reuses_fixed_node_without_endpoint_handle():
    app = QApplication.instance() or QApplication([])
    fixed_node = SimpleNamespace(
        index=10, position=QPointF(0.0, 2.0),
        xx=np.array([[0.0, -2.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]]),
    )
    inner_nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(3)
    ]
    main_edges = [SimpleNamespace(uv_index=1), SimpleNamespace(uv_index=1)]
    patch = grid_editor5.extended_patch(inner_nodes, main_edges)
    patch.one_cap_topology = SimpleNamespace(
        outer_start_node=fixed_node, outer_end_node=None
    )

    patch.enable_bezier_mode()

    roles = {handle.role for handle in patch.bezier_handles}
    assert "start" not in roles
    assert "start_tangent" not in roles
    assert roles == {"end", "end_tangent"}
    assert patch.outer_nodes[0] is fixed_node
    assert len(patch.outer_nodes) == patch.required_outer_node_count
    assert patch.bezier_start_position() is fixed_node.position
    assert np.allclose(patch.bezier_start_vector(), [2.0, 0.0])
    original_position = QPointF(fixed_node.position)
    end_tangent = next(
        handle for handle in patch.bezier_handles
        if handle.role == "end_tangent"
    )
    end_tangent.move_to_scene(end_tangent.pos() + QPointF(0.0, 1.0))
    assert fixed_node.position == original_position
    assert patch.outer_nodes[0] is fixed_node
    assert app is not None


def test_two_cap_bezier_has_no_endpoint_or_tangent_handles():
    app = QApplication.instance() or QApplication([])
    inner_nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(3)
    ]
    start_node = SimpleNamespace(
        index=10, position=QPointF(0.0, 2.0),
        xx=np.array([[0.0, -2.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]]),
    )
    end_node = SimpleNamespace(
        index=11, position=QPointF(2.0, 2.0),
        xx=np.array([[0.0, 3.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]]),
    )
    patch = grid_editor5.extended_patch(
        inner_nodes,
        [SimpleNamespace(uv_index=1), SimpleNamespace(uv_index=1)],
    )
    patch.capped_gap = SimpleNamespace(
        outer_start_node=start_node, outer_end_node=end_node
    )
    patch.set_outer_positions([
        start_node.position, QPointF(1.0, 2.0), end_node.position
    ])

    patch.enable_bezier_mode()

    assert patch.bezier_handles == []
    assert patch.outer_nodes[0] is start_node
    assert patch.outer_nodes[-1] is end_node
    assert np.allclose(patch.bezier_start_vector(), [2.0, 0.0])
    assert np.allclose(patch.bezier_end_vector(), [-3.0, 0.0])
    assert app is not None


def test_one_cap_commit_rejects_incomplete_outer_row(capsys):
    inner_nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(3)
    ]
    patch = grid_editor5.extended_patch(inner_nodes, [object(), object()])
    patch.one_cap_topology = SimpleNamespace()

    result = grid_editor5.add_one_cap_gap_to_nodes_elements(patch)

    assert result is None
    assert "requires exactly 3 outer preview nodes" in capsys.readouterr().out


@pytest.mark.parametrize(
    "click_order",
    [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [2, 0, 3, 1],
        [1, 3, 0, 2],
    ],
)
def test_extended_outer_nodes_are_ordered_independently_of_clicks(click_order):
    app = QApplication.instance() or QApplication([])
    inner_nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(4)
    ]
    ordered_edges = [object(), object(), object()]
    outer_positions = [
        QPointF(float(index), 10.0) for index in range(4)
    ]
    patch = grid_editor5.extended_patch(inner_nodes, ordered_edges)

    for index in click_order:
        assert patch.add_outer_node(outer_positions[index])

    assert [node.position for node in patch.outer_nodes] == outer_positions
    assert app is not None


def test_bezier_outer_nodes_follow_inner_arc_length_fractions():
    app = QApplication.instance() or QApplication([])
    inner_nodes = [
        SimpleNamespace(index=0, position=QPointF(0.0, 0.0)),
        SimpleNamespace(index=1, position=QPointF(1.0, 0.0)),
        SimpleNamespace(index=2, position=QPointF(4.0, 0.0)),
    ]
    patch = grid_editor5.extended_patch(inner_nodes, [object(), object()])
    patch.set_outer_positions([
        QPointF(0.0, 2.0), QPointF(1.0, 2.0), QPointF(4.0, 2.0)
    ])

    patch.enable_bezier_mode()

    assert len(patch.outer_nodes) == 3
    assert len(patch.outer_tangents) == 3
    assert np.allclose(patch.outer_parameters, [0.0, 0.25, 1.0], atol=2e-3)
    assert np.allclose(
        [[node.position.x(), node.position.y()] for node in patch.outer_nodes],
        [[0.0, 2.0], [1.0, 2.0], [4.0, 2.0]], atol=2e-3,
    )
    assert all(np.linalg.norm(tangent) > 0 for tangent in patch.outer_tangents)
    assert app is not None


def test_dragging_bezier_handles_updates_outer_preview():
    app = QApplication.instance() or QApplication([])
    inner_nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(3)
    ]
    patch = grid_editor5.extended_patch(inner_nodes, [object(), object()])
    patch.set_outer_positions([
        QPointF(0.0, 2.0), QPointF(1.0, 2.0), QPointF(2.0, 2.0)
    ])
    patch.enable_bezier_mode()
    handles = {handle.role: handle for handle in patch.bezier_handles}
    old_middle = patch.outer_nodes[1].position
    old_end_vector = (
        handles["end_tangent"].pos() - handles["end"].pos()
    )

    handles["start_tangent"].move_to_scene(QPointF(0.0, 5.0))
    assert patch.outer_nodes[1].position != old_middle
    assert len(patch.outer_tangents) == 3

    handles["end"].move_to_scene(QPointF(3.0, 3.0))
    assert patch.outer_nodes[-1].position == QPointF(3.0, 3.0)
    assert (
        handles["end_tangent"].pos() - handles["end"].pos()
    ) == old_end_vector
    assert app is not None


def test_extended_bezier_handles_use_fixed_screen_size_and_cosmetic_outline():
    app = QApplication.instance() or QApplication([])
    inner_nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(3)
    ]
    patch = grid_editor5.extended_patch(inner_nodes, [object(), object()])
    patch.set_outer_positions([
        QPointF(0.0, 2.0), QPointF(1.0, 2.0), QPointF(2.0, 2.0)
    ])
    patch.enable_bezier_mode()

    assert patch.bezier_handles
    for handle in patch.bezier_handles:
        assert handle.flags() & grid_editor5.QGraphicsItem.ItemIgnoresTransformations
        assert handle.rect().width() == pytest.approx(
            grid_editor5.EXTENDED_BEZIER_HANDLE_SIZE
        )
        assert handle.rect().height() == pytest.approx(
            grid_editor5.EXTENDED_BEZIER_HANDLE_SIZE
        )
        assert handle.pen().isCosmetic()
        assert handle.pen().widthF() == pytest.approx(
            grid_editor5.GRAPHICS_HANDLE_OUTLINE_WIDTH
        )
    assert app is not None


@pytest.mark.parametrize("zoom_level", [0.1, 1.0, 10.0])
def test_extended_patch_paints_no_passive_node_markers(zoom_level):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    view.zoom_level = zoom_level
    inner_nodes = [
        SimpleNamespace(index=index, position=QPointF(float(index), 0.0))
        for index in range(3)
    ]
    patch = grid_editor5.extended_patch(inner_nodes, [object(), object()])
    patch.set_outer_positions([
        QPointF(0.0, 2.0), QPointF(1.0, 2.0), QPointF(2.0, 2.0)
    ])
    patch.set_radial_layers(2)
    patch.enable_bezier_mode()
    scene.addItem(patch)

    class RecordingPainter:
        def __init__(self):
            self.pen = None
            self.ellipses = []

        def setPen(self, pen):
            self.pen = pen

        def drawPath(self, unused_path):
            pass

        def drawEllipse(self, x, y, width, height):
            self.ellipses.append((x, y, width, height))

    painter = RecordingPainter()
    patch.paint(painter, None)

    assert patch.pen().isCosmetic()
    assert patch.pen().widthF() == pytest.approx(
        grid_editor5.EXTENDED_PATCH_LINE_WIDTH
    )
    assert painter.ellipses == []
    assert len(patch.outer_nodes) == 3
    assert len(patch.preview_node_rows) == 3
    assert patch.bezier_handles
    assert app is not None


def test_initial_bezier_midpoint_is_outside_owning_element(monkeypatch):
    app = QApplication.instance() or QApplication([])
    positions = [
        QPointF(0.0, 0.0), QPointF(2.0, 0.0),
        QPointF(2.0, 1.0), QPointF(0.0, 1.0),
    ]
    nodes = [
        SimpleNamespace(index=index, position=position)
        for index, position in enumerate(positions)
    ]
    owner = SimpleNamespace(index=7, vertices=[0, 1, 2, 3])
    edge = SimpleNamespace(
        nodes=[nodes[1], nodes[0]], vertices=[1, 0],
        uv_index=1, element_index=7,
    )
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [owner], raising=False)
    patch = grid_editor5.extended_patch([nodes[0], nodes[1]], [edge])

    patch.enable_bezier_mode()

    handles = {handle.role: handle for handle in patch.bezier_handles}
    start = np.array([handles["start"].pos().x(), handles["start"].pos().y()])
    end = np.array([handles["end"].pos().x(), handles["end"].pos().y()])
    start_vector = np.array([
        (handles["start_tangent"].pos() - handles["start"].pos()).x(),
        (handles["start_tangent"].pos() - handles["start"].pos()).y(),
    ])
    end_vector = np.array([
        (handles["end_tangent"].pos() - handles["end"].pos()).x(),
        (handles["end_tangent"].pos() - handles["end"].pos()).y(),
    ])
    curve_midpoint, unused_tangent = grid_editor5.cubic_bezier_point_and_tangent(
        start, end, start_vector, end_vector, 0.5
    )
    boundary_midpoint = np.array([1.0, 0.0])
    element_centroid = np.array([1.0, 0.5])
    assert np.inner(
        curve_midpoint - boundary_midpoint,
        element_centroid - boundary_midpoint,
    ) < 0
    assert np.allclose(curve_midpoint, [1.0, -1.0])
    assert app is not None


def test_imported_transverse_scale_is_inherited_at_old_boundary(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    positions = [
        (0.0, 0.0), (1.0, 0.0),
        (1.0, -0.3), (0.0, -0.3),
    ]
    nodes_xx = np.zeros((2, 4, len(positions)))
    for index, position in enumerate(positions):
        nodes_xx[:, 0, index] = position
        nodes_xx[:, 1, index] = [1.0 / 3.0, 0.0]
        nodes_xx[:, 2, index] = [0.0, 100.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 2)
        for index in range(len(positions))
    ]
    for node in nodes:
        scene.addItem(node)

    old_sizes = np.ones((4, 4))
    old_sizes[1, :] = [1.0, -1.0, 1.0, -1.0]
    old_sizes[2, :] = -0.001
    old_sizes[3, :] = old_sizes[1, :] * old_sizes[2, :]
    old_element = grid_editor5.mesh_element_record(
        10, [0, 1, 2, 3], old_sizes
    )
    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx)
    )
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    inner_edge_sizes = np.ones((2, 2))
    inner_edge_sizes[1, :] = old_sizes[1, :2]
    inner_edge = grid_editor5.boundary_edge(
        [nodes[0], nodes[1]], [0, 1], [0, 1],
        old_element.index, 0, 1, inner_edge_sizes,
    )
    scene.addItem(inner_edge)

    monkeypatch.setattr(
        grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx)
    )
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(
        grid_editor5, "element_list", [old_element], raising=False
    )
    monkeypatch.setattr(
        grid_editor5, "boundary_list", [inner_edge], raising=False
    )
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)
    monkeypatch.setattr(grid_editor5, "static_mesh", None, raising=False)

    old_vertices_snapshot = np.array(old_element.vertices, copy=True)
    old_sizes_snapshot = np.array(old_element.sizes, copy=True)
    old_raw_snapshot = np.array(nodes_xx, copy=True)
    old_node_count = nodes_xx.shape[2]
    old_controls_snapshot = element_bezier_points(
        grid_editor5.scaled_element_points(
            old_element.vertices, old_element.sizes
        ),
        1.0,
    )

    patch = grid_editor5.extended_patch(nodes[:2], [inner_edge])
    patch.set_outer_positions([QPointF(0.0, 0.3), QPointF(1.0, 0.3)])
    patch.enable_bezier_mode()
    scene.addItem(patch)
    view.current_extended_patch = patch
    view.selected_edges = [inner_edge]
    view.editable_depth = 2
    commit_result = {}
    real_commit = grid_editor5.add_extended_patch_to_nodes_elements

    def capture_commit(committed_patch):
        commit_result["value"] = real_commit(committed_patch)
        return commit_result["value"]

    monkeypatch.setattr(
        grid_editor5, "add_extended_patch_to_nodes_elements", capture_commit
    )

    assert view.commit_current_patch()
    node_rows, created_elements = commit_result["value"]

    assert len(created_elements) == 1
    new_element = created_elements[0]
    radial_uv = 2
    outer_by_inner = dict(zip(nodes[:2], node_rows[1]))
    for old_node in nodes[:2]:
        old_local = list(old_element.vertices).index(old_node.index)
        new_local = list(new_element.vertices).index(old_node.index)
        outer_node = outer_by_inner[old_node]
        radial_chord = (
            grid_editor5.np_point(outer_node.position)
            - grid_editor5.np_point(old_node.position)
        )
        new_size = new_element.sizes[radial_uv, new_local]
        effective = new_size * old_node.xx[:, radial_uv]
        assert abs(new_size) == pytest.approx(
            abs(old_element.sizes[radial_uv, old_local])
        )
        assert abs(new_size) != pytest.approx(1.0)
        assert np.inner(effective, radial_chord) > 0.0
        assert np.linalg.norm(effective) / np.linalg.norm(
            radial_chord
        ) == pytest.approx(1.0 / 3.0)
        assert new_element.sizes[3, new_local] == pytest.approx(
            new_element.sizes[1, new_local]
            * new_element.sizes[2, new_local]
        )

        outer_local = list(new_element.vertices).index(outer_node.index)
        assert abs(new_element.sizes[radial_uv, outer_local]) == pytest.approx(
            1.0
        )

    assert np.array_equal(old_element.vertices, old_vertices_snapshot)
    assert np.array_equal(old_element.sizes, old_sizes_snapshot)
    assert np.array_equal(
        grid_editor5.jorek.nodes_xx[:, :, :old_node_count], old_raw_snapshot
    )
    assert np.array_equal(
        element_bezier_points(
            grid_editor5.scaled_element_points(
                old_element.vertices, old_element.sizes
            ),
            1.0,
        ),
        old_controls_snapshot,
    )
    assert view.editable_depth == 2
    assert view._element_adjacency is not None
    assert view.editable_element_indices_set == {10, 11}
    assert app is not None


@pytest.mark.parametrize(
    "edge_count, radial_layers, bezier_mode, main_uv_index",
    [
        (2, 1, False, 1), (4, 1, False, 1),
        (2, 2, False, 1), (2, 4, False, 1),
        (2, 1, True, 1), (2, 4, True, 1), (2, 1, True, 2),
    ],
)
def test_extended_patch_creates_radial_rows(
    monkeypatch, edge_count, radial_layers, bezier_mode, main_uv_index, capsys
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    inner_count = edge_count + 1
    nodes_xx = np.zeros((2, 4, inner_count))
    radial_uv_index = main_uv_index % 2 + 1
    for index in range(inner_count):
        nodes_xx[:,0,index] = [float(index), 0.0]
        nodes_xx[:,main_uv_index,index] = [1.0, 0.0]
        nodes_xx[:,radial_uv_index,index] = [0.0, 1.0 / 3.0]
    inner_nodes = [
        jorek_node_item(index, nodes_xx[:,:,index], 2)
        for index in range(inner_count)
    ]
    for node in inner_nodes:
        scene.addItem(node)

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", inner_nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)

    inner_edges = []
    for index in range(edge_count):
        edge_nodes = [inner_nodes[index], inner_nodes[index + 1]]
        edge = grid_editor5.boundary_edge(
            edge_nodes, [index, index + 1], [0,1], 100 + index, 0,
            main_uv_index,
            grid_editor5.signed_edge_sizes(edge_nodes, main_uv_index),
        )
        inner_edges.append(edge)
        scene.addItem(edge)
    monkeypatch.setattr(
        grid_editor5, "boundary_list", list(inner_edges), raising=False
    )

    patch = grid_editor5.extended_patch(inner_nodes, inner_edges)
    outer_positions = [
        QPointF(float(index), 1.0) for index in range(inner_count)
    ]
    for index in reversed(range(inner_count)):
        patch.add_outer_node(outer_positions[index])
    patch.set_radial_layers(radial_layers)
    if bezier_mode:
        patch.enable_bezier_mode()
    install_edge_owner_lookup(monkeypatch, inner_edges)
    preview_final_positions = [
        QPointF(node.position) for node in patch.preview_node_rows[-1]
    ]
    scene.addItem(patch)
    view.current_extended_patch = patch
    view.selected_edges = list(inner_edges)
    old_node_count = len(grid_editor5.node_list)
    helper_results = []
    real_one_edge_helper = grid_editor5.add_element_from_one_edge
    real_two_edge_helper = grid_editor5.add_element_from_two_edges

    def record_one_edge(*args, **kwargs):
        element = real_one_edge_helper(*args, **kwargs)
        helper_results.append(("one", element))
        return element

    def record_two_edges(*args, **kwargs):
        element = real_two_edge_helper(*args, **kwargs)
        helper_results.append(("two", element))
        return element

    monkeypatch.setattr(grid_editor5, "add_element_from_one_edge", record_one_edge)
    monkeypatch.setattr(grid_editor5, "add_element_from_two_edges", record_two_edges)

    if bezier_mode:
        commit_result = {}
        real_commit = grid_editor5.add_extended_patch_to_nodes_elements

        def capture_commit(committed_patch):
            commit_result["value"] = real_commit(committed_patch)
            return commit_result["value"]

        monkeypatch.setattr(
            grid_editor5, "add_extended_patch_to_nodes_elements",
            capture_commit,
        )
        event = type("KeyEvent", (), {"key": lambda self: Qt.Key_P})()
        view.keyPressEvent(event)
        result = commit_result["value"]
    else:
        result = grid_editor5.add_extended_patch_to_nodes_elements(patch)
    diagnostic_output = capsys.readouterr().out

    node_rows, elements = result
    assert [kind for kind, element in helper_results] == (
        (["one"] + ["two"] * (edge_count - 1)) * radial_layers
    )
    assert [element for kind, element in helper_results] == elements
    assert len(grid_editor5.node_list) - old_node_count == inner_count * radial_layers
    assert len(node_rows) == radial_layers + 1
    assert all(len(row) == inner_count for row in node_rows)
    for row in node_rows:
        for node in row:
            assert_basis_handles_match_vectors(node)
    if bezier_mode:
        assert all(
            np.allclose(
                node.xx[:, main_uv_index],
                patch.preview_along_vectors[radial_index][column],
            )
            for radial_index, row in enumerate(node_rows)
            for column, node in enumerate(row)
        )
    assert all(
        np.linalg.norm(
            grid_editor5.np_point(created_node.position)
            - grid_editor5.np_point(preview_node.position)
        ) == pytest.approx(0.0)
        for created_row, preview_row in zip(
            node_rows[1:], patch.preview_node_rows[1:]
        )
        for created_node, preview_node in zip(created_row, preview_row)
    )
    assert len({node.index for row in node_rows for node in row}) == (
        inner_count * (radial_layers + 1)
    )
    if radial_layers > 1:
        assert diagnostic_output.count(
            "main edge uv_index = " + str(main_uv_index)
        ) == (
            inner_count * radial_layers
        )
        assert diagnostic_output.count(
            "perp_index = " + str(radial_uv_index)
        ) == (
            inner_count * radial_layers
        )
        assert diagnostic_output.count("dot after =") == (
            inner_count * radial_layers
        )
        assert all(
            np.dot(
                new_node.xx[:, 2],
                node_rows[0][column].xx[:, 2],
            ) > 0.0
            for radial_row in node_rows[1:]
            for column, new_node in enumerate(radial_row)
        )
        assert all(
            np.array_equal(
                node.xx[:, radial_uv_index],
                grid_editor5.this_scaling
                * grid_editor5.jorek.nodes_xx[:, radial_uv_index, node.index],
            )
            for row in node_rows[1:]
            for node in row
        )
    assert [node.position for node in node_rows[-1]] == preview_final_positions
    for outer_node in node_rows[-1]:
        assert np.allclose(outer_node.xx[:,3], [0.0, 0.0])
    assert len(elements) == edge_count * radial_layers
    assert len(grid_editor5.element_list) == edge_count * radial_layers
    assert all(edge not in grid_editor5.boundary_list for edge in inner_edges)
    assert all(edge.scene() is scene for edge in inner_edges)
    assert all(not edge.active and not edge.isVisible() for edge in inner_edges)
    outer_node_indices = {node.index for node in node_rows[-1]}
    outer_edges = [
        edge for edge in grid_editor5.boundary_list
        if set(edge.vertices).issubset(outer_node_indices)
    ]
    interface_index_sets = [
        {node.index for node in row} for row in node_rows[1:-1]
    ]
    intermediate_row_edges = [
        edge for edge in grid_editor5.boundary_list
        if any(set(edge.vertices).issubset(indices) for indices in interface_index_sets)
    ]
    assert intermediate_row_edges == []
    assert len(outer_edges) == edge_count
    assert len(grid_editor5.boundary_list) == edge_count + 2 * radial_layers
    created_edges = {
        edge for element in elements for edge in element.edges
    }
    all_transverse_edges = [
        edge for edge in created_edges
        if len(set(edge.vertices).intersection(outer_node_indices)) == 1
    ]
    internal_transverse_edges = [
        edge for edge in all_transverse_edges
        if edge not in grid_editor5.boundary_list
    ]
    assert len(internal_transverse_edges) == edge_count - 1
    for radial_index in range(radial_layers):
        inner_indices = [node.index for node in node_rows[radial_index]]
        next_indices = [node.index for node in node_rows[radial_index + 1]]
        layer_elements = elements[
            radial_index * edge_count:(radial_index + 1) * edge_count
        ]
        for index, element in enumerate(layer_elements):
            expected_vertices = [
                inner_indices[index], inner_indices[index + 1],
                next_indices[index + 1], next_indices[index],
            ]
            assert set(element.vertices) == set(expected_vertices)
            assert all(
                frozenset((
                    element.vertices[side], element.vertices[(side + 1) % 4]
                )) in {
                    frozenset((
                        expected_vertices[expected_side],
                        expected_vertices[(expected_side + 1) % 4],
                    ))
                    for expected_side in range(4)
                }
                for side in range(4)
            )
    if main_uv_index == 1:
        for index, element in enumerate(elements[:edge_count]):
            control_net = element_bezier_points(element.points, 1.0)
            right_edge = control_net[:,3,:]
            left_edge = control_net[:,0,:]
            assert np.allclose(right_edge[0,:], float(index + 1))
            assert np.allclose(left_edge[0,:], float(index))
            layer_height = 1.0 / radial_layers
            assert right_edge[1,0] == pytest.approx(0.0)
            assert right_edge[1,-1] == pytest.approx(layer_height)
            assert left_edge[1,0] == pytest.approx(0.0)
            assert left_edge[1,-1] == pytest.approx(layer_height)
            outer_edge = control_net[:,:,3]
            if not (bezier_mode and radial_layers > 1):
                assert np.all(np.diff(outer_edge[0,:]) >= 0)
            assert np.allclose(outer_edge[1,:], layer_height)
    assert all(edge.uv_index == main_uv_index for edge in outer_edges)
    assert all(edge.uv_index == radial_uv_index for edge in all_transverse_edges)
    if bezier_mode:
        assert diagnostic_output.count("Bezier outer edge") == edge_count
        nodal_scales = grid_editor5.bezier_nodal_parameter_scales(
            patch.outer_parameters
        )
        for node, tangent, scale in zip(
            node_rows[-1], patch.outer_tangents, nodal_scales
        ):
            assert np.allclose(
                node.xx[:, main_uv_index], scale * np.asarray(tangent) / 3.0
            )
            assert np.dot(
                node.xx[:, main_uv_index], np.asarray(tangent)
            ) > 0.0
            assert_basis_handles_match_vectors(node)
        neighbouring_magnitudes = [
            np.linalg.norm(node.xx[:, main_uv_index]) for node in node_rows[-2]
        ]
        outer_magnitudes = [
            np.linalg.norm(node.xx[:, main_uv_index]) for node in node_rows[-1]
        ]
        assert all(
            0.25 <= outer / neighbouring <= 4.0
            for outer, neighbouring in zip(
                outer_magnitudes, neighbouring_magnitudes
            )
        )
        final_indices = [node.index for node in node_rows[-1]]
        for index in range(edge_count):
            edge = next(
                edge for edge in outer_edges
                if set(edge.vertices) == set(final_indices[index:index + 2])
            )
            controls = edge_bezier_points(edge.points)
            controls = np.column_stack((
                controls[:, 0, 0], controls[:, 1, 0],
                controls[:, 1, 1], controls[:, 0, 1],
            ))
            if edge.vertices[0] != final_indices[index]:
                controls = controls[:, ::-1]
            dt = patch.outer_parameters[index + 1] - patch.outer_parameters[index]
            expected_sizes = (
                np.array([
                    dt / nodal_scales[index],
                    -dt / nodal_scales[index + 1],
                ])
                if edge.vertices[0] == final_indices[index]
                else np.array([
                    -dt / nodal_scales[index + 1],
                    dt / nodal_scales[index],
                ])
            )
            assert np.allclose(edge.sizes[1, :], expected_sizes)
            expected_effective_by_node = {
                final_indices[index]: (
                    dt * np.asarray(patch.outer_tangents[index]) / 3.0
                ),
                final_indices[index + 1]: (
                    -dt * np.asarray(patch.outer_tangents[index + 1]) / 3.0
                ),
            }
            for endpoint, node in enumerate(edge.nodes):
                assert np.allclose(
                    node.xx[:, main_uv_index] * edge.sizes[1, endpoint],
                    expected_effective_by_node[node.index],
                )
            expected_controls = np.column_stack((
                grid_editor5.np_point(patch.outer_nodes[index].position),
                grid_editor5.np_point(patch.outer_nodes[index].position)
                + dt * np.asarray(patch.outer_tangents[index]) / 3.0,
                grid_editor5.np_point(patch.outer_nodes[index + 1].position)
                - dt * np.asarray(patch.outer_tangents[index + 1]) / 3.0,
                grid_editor5.np_point(patch.outer_nodes[index + 1].position),
            ))
            assert np.allclose(controls, expected_controls, atol=1e-10)
    assert_boundary_status_matches_edges(
        grid_editor5.node_list, grid_editor5.boundary_list
    )
    assert view.current_extended_patch is None
    assert patch.scene() is None
    assert app is not None


def test_capped_two_element_gap_reuses_sequential_single_element_helpers(
    monkeypatch
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    positions = [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        (0.0, 1.0), (2.0, 1.0),
    ]
    nodes_xx = np.zeros((2, 4, len(positions)))
    for index, position in enumerate(positions):
        nodes_xx[:,0,index] = position
        nodes_xx[:,1,index] = [1.0 / 3.0, 0.0]
        nodes_xx[:,2,index] = [0.0, 1.0 / 3.0]
    nodes_xx[:,1,3] = [0.5, 1.0]
    nodes_xx[:,1,4] = [0.5, -1.0]
    nodes_xx[:,2,3:] = [[0.0, 0.0], [-1.0 / 3.0, -1.0 / 3.0]]
    nodes = [
        jorek_node_item(index, nodes_xx[:,:,index], 2)
        for index in range(len(positions))
    ]
    for node in nodes:
        scene.addItem(node)

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)

    def edge(start, end, uv_index, element_index):
        edge_nodes = [nodes[start], nodes[end]]
        result = grid_editor5.boundary_edge(
            edge_nodes, [start, end], [0,1], element_index, 0, uv_index,
            grid_editor5.signed_edge_sizes(edge_nodes, uv_index),
        )
        scene.addItem(result)
        return result

    inner_edges = [edge(0, 1, 1, 10), edge(1, 2, 1, 11)]
    start_side = edge(0, 3, 2, 12)
    end_side = edge(2, 4, 2, 13)
    selected_edges = [end_side, inner_edges[1], start_side, inner_edges[0]]
    install_edge_owner_lookup(monkeypatch, selected_edges)
    monkeypatch.setattr(
        grid_editor5, "boundary_list", list(selected_edges), raising=False
    )
    gap = grid_editor5.ordered_capped_boundary_gap(selected_edges)
    patch = grid_editor5.extended_patch(
        gap.inner_nodes, gap.inner_edges, can_commit=False
    )
    patch.capped_gap = gap
    patch.set_outer_positions([
        nodes[3].position, QPointF(1.0, 1.0), nodes[4].position
    ])
    straight_middle = patch.outer_nodes[1].position
    patch.enable_bezier_mode()
    assert patch.bezier_handles == []
    assert patch.path().elementCount() == 13
    assert patch.outer_nodes[0] is nodes[3]
    assert patch.outer_nodes[-1] is nodes[4]
    assert patch.outer_nodes[1].position != straight_middle
    bezier_middle = patch.outer_nodes[1].position
    scene.addItem(patch)
    view.current_extended_patch = patch
    view.selected_edges = list(selected_edges)

    helper_calls = []
    real_two_edges = grid_editor5.add_element_from_two_edges
    real_three_edges = grid_editor5.add_element_from_three_edges

    def record_two_edges(*args, **kwargs):
        element = real_two_edges(*args, **kwargs)
        helper_calls.append(("two", element))
        return element

    def record_three_edges(*args, **kwargs):
        element = real_three_edges(*args, **kwargs)
        helper_calls.append(("three", element))
        return element

    monkeypatch.setattr(grid_editor5, "add_element_from_two_edges", record_two_edges)
    monkeypatch.setattr(
        grid_editor5, "add_element_from_three_edges", record_three_edges
    )
    old_node_count = len(nodes)

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(patch)

    assert [kind for kind, element in helper_calls] == ["two", "three"]
    assert [element for kind, element in helper_calls] == elements
    assert len(grid_editor5.node_list) == old_node_count + 1
    assert node_rows[1][0] is nodes[3]
    assert node_rows[1][-1] is nodes[4]
    assert node_rows[1][1].position == bezier_middle
    assert [element.vertices for element in elements] == [
        [0, 1, node_rows[1][1].index, 3],
        [1, 2, 4, node_rows[1][1].index],
    ]
    assert all(edge not in grid_editor5.boundary_list for edge in selected_edges)
    assert len(grid_editor5.boundary_list) == 2
    assert {
        frozenset(edge.vertices) for edge in grid_editor5.boundary_list
    } == {
        frozenset((3, node_rows[1][1].index)),
        frozenset((node_rows[1][1].index, 4)),
    }
    assert_boundary_status_matches_edges(nodes + [node_rows[1][1]], grid_editor5.boundary_list)
    assert view.current_extended_patch is None
    assert patch.scene() is None
    assert app is not None


def build_two_cap_selection(
    monkeypatch, radial_layers, main_edge_count=3, end_radial_layers=None,
    main_uv_index=1,
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    if end_radial_layers is None:
        end_radial_layers = radial_layers
    radial_uv_index = main_uv_index % 2 + 1
    positions = [
        (float(column), 0.0) for column in range(main_edge_count + 1)
    ]
    start_indices = [0]
    for radial_index in range(1, radial_layers + 1):
        start_indices.append(len(positions))
        positions.append((0.0, float(radial_index)))
    end_indices = [main_edge_count]
    for radial_index in range(1, end_radial_layers + 1):
        end_indices.append(len(positions))
        positions.append((float(main_edge_count), float(radial_index)))

    nodes_xx = np.zeros((2, 4, len(positions)))
    for index, position in enumerate(positions):
        nodes_xx[:, 0, index] = position
        nodes_xx[:, main_uv_index, index] = [1.0 / 3.0, 0.0]
        nodes_xx[:, radial_uv_index, index] = [0.0, 1.0 / 3.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 2)
        for index in range(len(positions))
    ]
    for node in nodes:
        scene.addItem(node)

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)

    def edge(start, end, uv_index, element_index):
        edge_nodes = [nodes[start], nodes[end]]
        result = grid_editor5.boundary_edge(
            edge_nodes, [start, end], [0, 1], element_index, 0, uv_index,
            grid_editor5.signed_edge_sizes(edge_nodes, uv_index),
        )
        scene.addItem(result)
        return result

    main_edges = [
        edge(column, column + 1, main_uv_index, 10 + column)
        for column in range(main_edge_count)
    ]
    start_cap_edges = [
        edge(
            start_indices[index], start_indices[index + 1],
            radial_uv_index, 30 + index,
        )
        for index in range(radial_layers)
    ]
    end_cap_edges = [
        edge(
            end_indices[index], end_indices[index + 1],
            radial_uv_index, 40 + index,
        )
        for index in range(end_radial_layers)
    ]
    selected_edges = list(reversed(
        main_edges + start_cap_edges + end_cap_edges
    ))
    owners = install_edge_owner_lookup(monkeypatch, selected_edges)
    monkeypatch.setattr(
        grid_editor5, "boundary_list", list(selected_edges), raising=False
    )
    view.selected_edges = list(selected_edges)
    return SimpleNamespace(
        app=app, scene=scene, view=view, nodes=nodes,
        main_edges=main_edges, start_cap_edges=start_cap_edges,
        end_cap_edges=end_cap_edges, selected_edges=selected_edges,
        start_nodes=[nodes[index] for index in start_indices],
        end_nodes=[nodes[index] for index in end_indices],
        main_uv_index=main_uv_index,
        owners=owners,
    )


def create_two_cap_patch(case):
    case.view.set_pending_bezier_mode(False)
    case.view.keyPressEvent(
        type("KeyEvent", (), {
            "key": lambda self: Qt.Key_0 + case.main_uv_index
        })()
    )
    case.view.keyPressEvent(
        type("KeyEvent", (), {"key": lambda self: Qt.Key_E})()
    )
    return case.view.current_extended_patch


def test_view_remains_usable_without_extended_patch_controls():
    app = QApplication.instance() or QApplication([])
    view = this_view()

    assert view.patch_controls is None
    assert view.pending_main_uv_index is None
    assert view.pending_radial_layers == 1
    view.set_pending_main_uv_index(2)
    view.set_extended_radial_layers(4)
    assert view.pending_main_uv_index == 2
    assert view.pending_radial_layers == 4
    assert app is not None


def test_extended_patch_panel_maps_direction_and_keyboard_state():
    app = QApplication.instance() or QApplication([])
    view = this_view()
    controls = grid_editor5.extended_patch_controls(view)

    controls.main_direction_combo.setCurrentIndex(1)
    assert view.pending_main_uv_index == 1
    controls.main_direction_combo.setCurrentIndex(2)
    assert view.pending_main_uv_index == 2
    controls.main_direction_combo.setCurrentIndex(0)
    assert view.pending_main_uv_index is None

    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_1})())
    assert view.pending_main_uv_index == 1
    assert controls.main_direction_combo.currentData() == 1
    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_2})())
    assert view.pending_main_uv_index == 2
    assert controls.main_direction_combo.currentData() == 2
    assert app is not None


def test_editable_depth_control_rebuilds_overlay_without_geometry_or_view_change(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    for name in (
        "jorek", "scene", "view", "node_list", "element_list",
        "boundary_list", "static_mesh",
    ):
        monkeypatch.setattr(
            grid_editor5, name, getattr(grid_editor5, name), raising=False
        )
    grid = editable_depth_grid()
    scene, unused_nodes, unused_elements, unused_edges = (
        grid_editor5.build_grid_scene(grid, 1.0)
    )
    view = this_view()
    view.resize(640, 480)
    view.setScene(scene)
    grid_editor5.view = view
    grid_editor5.rebuild_graphics_layers(active_view=view)
    controls = grid_editor5.extended_patch_controls(view)
    app.processEvents()

    boundary_keys_before = {
        frozenset(edge.vertices) for edge in grid_editor5.boundary_list
    }
    static_path_count = grid_editor5.static_mesh.path().elementCount()
    nodes_xx_before = np.array(grid.nodes_xx, copy=True)
    vertices_before = np.array(grid.vertices, copy=True)
    sizes_before = np.array(grid.elements_size, copy=True)
    view.setTransform(QTransform().scale(2.25, 2.25))
    view.centerOn(QPointF(2.0, 2.0))
    app.processEvents()
    transform_before = QTransform(view.transform())
    center_before = view.mapToScene(view.viewport().rect().center())
    adjacency_before = view._element_adjacency
    preview = QGraphicsPathItem()
    scene.addItem(preview)
    view.current_patch = preview
    view.selected_edges = [grid_editor5.boundary_list[0]]
    editable_node = next(
        node for node in grid_editor5.node_list
        if isinstance(node, grid_editor5.jorek_node_item)
    )
    view.selected_point = editable_node.blue_handle
    view.dragged_node = editable_node

    controls.editable_depth_spin.setValue(1)
    for node in grid_editor5.node_list:
        if not isinstance(node, grid_editor5.jorek_node_item):
            continue
        for handle in (node.blue_handle, node.red_handle):
            assert (
                handle.flags()
                & grid_editor5.QGraphicsItem.ItemIgnoresTransformations
            )
            assert handle.rect().width() == pytest.approx(
                grid_editor5.VECTOR_HANDLE_SIZE
            )
            assert handle.rect().height() == pytest.approx(
                grid_editor5.VECTOR_HANDLE_SIZE
            )
    app.processEvents()

    expected_elements = grid_editor5.editable_element_indices(
        grid_editor5.element_list,
        grid_editor5.boundary_list,
        depth=1,
        adjacency=view._element_adjacency,
    )
    expected_nodes = grid_editor5.editable_node_indices(
        grid_editor5.element_list, expected_elements
    )
    assert view.editable_depth == 1
    assert controls.editable_depth_spin.value() == 1
    assert view._element_adjacency is adjacency_before
    assert view.editable_element_indices_set == expected_elements
    assert view.editable_node_indices_set == expected_nodes
    assert view.current_patch is None
    assert view.current_extended_patch is None
    assert view.selected_edges == []
    assert view.selected_point is None
    assert view.dragged_node is None
    assert {
        element.index for element in grid_editor5.element_list
        if isinstance(element, grid_editor5.jorek_element_item)
    } == expected_elements
    assert {
        node.index for node in grid_editor5.node_list
        if isinstance(node, grid_editor5.jorek_node_item)
    } == expected_nodes
    assert {
        frozenset(edge.vertices) for edge in grid_editor5.boundary_list
    } == boundary_keys_before
    assert {
        edge.element_index for edge in grid_editor5.boundary_list
    } == grid_editor5.editable_boundary_element_indices(
        grid_editor5.element_list, grid_editor5.boundary_list
    )
    assert grid_editor5.static_mesh.path().elementCount() == static_path_count
    assert np.array_equal(grid.nodes_xx, nodes_xx_before)
    assert np.array_equal(grid.vertices, vertices_before)
    assert np.array_equal(grid.elements_size, sizes_before)
    assert view.transform() == transform_before
    center_after = view.mapToScene(view.viewport().rect().center())
    assert center_after.x() == pytest.approx(center_before.x(), abs=0.5)
    assert center_after.y() == pytest.approx(center_before.y(), abs=0.5)
    assert "depth 1" in controls.editable_region_label.text()
    assert app is not None


def test_extended_patch_panel_updates_uncapped_and_capped_radial_layers():
    app = QApplication.instance() or QApplication([])
    view = this_view()
    controls = grid_editor5.extended_patch_controls(view)

    class FakePatch:
        def __init__(self, topology=None, radial_layers=1):
            self.ordered_edges = [SimpleNamespace(uv_index=1)] * 3
            self.one_cap_topology = topology
            self.capped_gap = None
            self.radial_layers = radial_layers
            self.bezier_mode = False

        def main_uv_index(self):
            return 1

        def set_radial_layers(self, value):
            self.radial_layers = value

    view.current_extended_patch = FakePatch()
    controls.update_from_view()
    assert controls.radial_layers_spin.isEnabled()
    controls.radial_layers_spin.setValue(4)
    assert view.current_extended_patch.radial_layers == 4
    view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_6})())
    assert view.current_extended_patch.radial_layers == 6
    assert controls.radial_layers_spin.value() == 6

    cap_topology = SimpleNamespace(
        start_cap_edges=[object(), object(), object()], end_cap_edges=[]
    )
    view.current_extended_patch = FakePatch(cap_topology, radial_layers=3)
    controls.update_from_view()
    assert controls.radial_layers_spin.value() == 3
    assert not controls.radial_layers_spin.isEnabled()
    assert "fixed by cap chain" in controls.radial_note_label.text()
    assert app is not None


def test_radial_layer_spinbox_accepts_multi_digit_values():
    app = QApplication.instance() or QApplication([])
    view = this_view()
    controls = grid_editor5.extended_patch_controls(view)

    assert controls.radial_layers_spin.minimum() == 1
    assert (
        controls.radial_layers_spin.maximum()
        == grid_editor5.MAX_RADIAL_LAYERS
    )
    for value in (12, 25, 50):
        editor = controls.radial_layers_spin.lineEdit()
        editor.setFocus()
        editor.selectAll()
        QTest.keyClicks(editor, str(value))
        QTest.keyClick(editor, Qt.Key_Return)
        app.processEvents()
        assert controls.radial_layers_spin.value() == value
        assert view.pending_radial_layers == value
    controls.close()


def test_extended_patch_panel_buttons_route_to_view_methods(monkeypatch):
    app = QApplication.instance() or QApplication([])
    view = this_view()
    controls = grid_editor5.extended_patch_controls(view)
    calls = []
    monkeypatch.setattr(
        view, "create_extended_patch_preview",
        lambda: calls.append("preview"),
    )
    monkeypatch.setattr(
        view, "commit_current_patch", lambda: calls.append("commit"),
    )
    monkeypatch.setattr(
        view, "cancel_current_operation", lambda: calls.append("cancel"),
    )
    for button in (
        controls.preview_button, controls.commit_button,
        controls.cancel_button,
    ):
        button.setEnabled(True)
        button.click()

    assert calls == ["preview", "commit", "cancel"]
    assert app is not None


def test_grid_editor_window_wraps_independently_constructed_view():
    app = QApplication.instance() or QApplication([])
    view = this_view()
    window = grid_editor5.grid_editor_window(view)

    assert window.view is view
    assert window.patch_controls.view is view
    assert view.patch_controls is window.patch_controls
    assert window.centralWidget() is not None
    assert window.patch_controls.width() == 250
    layout = window.centralWidget().layout()
    assert layout.itemAt(0).widget() is window.patch_controls
    assert layout.itemAt(1).widget() is window.view
    window.close()
    assert app is not None


def test_window_commands_work_once_from_each_child_focus(monkeypatch):
    app = QApplication.instance() or QApplication([])
    view = this_view()
    calls = []
    command_methods = {
        "fit_grid_to_window": "fit",
        "create_extended_patch_preview": "preview",
        "enable_current_patch_bezier": "bezier",
        "commit_current_patch": "commit",
        "reset_zoom": "reset_zoom",
        "cancel_current_operation": "cancel",
    }
    for method_name, command_name in command_methods.items():
        monkeypatch.setattr(
            view, method_name,
            lambda name=command_name: calls.append(name),
        )

    window = grid_editor5.grid_editor_window(view)
    window.resize(900, 600)
    window.show()
    app.processEvents()
    focus_widgets = (
        view,
        window.patch_controls.radial_layers_spin,
        window.patch_controls.outer_boundary_combo,
        window.patch_controls.preview_button,
        window.patch_controls.fit_button,
    )
    commands = (
        (Qt.Key_F, "fit"),
        (Qt.Key_E, "preview"),
        (Qt.Key_B, "bezier"),
        (Qt.Key_P, "commit"),
        (Qt.Key_U, "reset_zoom"),
        (Qt.Key_Escape, "cancel"),
    )
    for key, command_name in commands:
        calls.clear()
        for widget in focus_widgets:
            widget.setFocus()
            app.processEvents()
            QTest.keyClick(widget, key)
            app.processEvents()
        assert calls == [command_name] * len(focus_widgets)

    window.close()
    assert app is not None


def test_window_numeric_shortcuts_do_not_steal_spinbox_entry():
    app = QApplication.instance() or QApplication([])
    window = grid_editor5.grid_editor_window()
    window.show()
    app.processEvents()
    spin = window.patch_controls.radial_layers_spin
    editor = spin.lineEdit()

    for value in (12, 25, 50):
        editor.setFocus()
        editor.selectAll()
        QTest.keyClicks(editor, str(value))
        QTest.keyClick(editor, Qt.Key_Return)
        app.processEvents()
        assert spin.value() == value
        assert window.view.pending_radial_layers == value

    window.close()


def test_fit_button_and_f_key_use_same_fit_path(monkeypatch):
    app = QApplication.instance() or QApplication([])
    view = this_view()
    controls = grid_editor5.extended_patch_controls(view)
    calls = []

    def fit():
        calls.append("fit")
        view.auto_fit_on_resize = True

    monkeypatch.setattr(view, "fit_grid_to_window", fit)
    view.auto_fit_on_resize = False
    controls.fit_button.click()
    assert calls == ["fit"]
    assert view.auto_fit_on_resize

    view.auto_fit_on_resize = False
    view.keyPressEvent(
        type("KeyEvent", (), {"key": lambda self: Qt.Key_F})()
    )
    assert calls == ["fit", "fit"]
    assert view.auto_fit_on_resize
    assert controls.fit_button.text() == "Fit to window"
    assert app is not None


def test_keyboard_bezier_action_refreshes_attached_panel(monkeypatch):
    case = build_two_cap_selection(monkeypatch, 1, main_edge_count=2)
    controls = grid_editor5.extended_patch_controls(case.view)
    controls.main_direction_combo.setCurrentIndex(1)
    controls.outer_boundary_combo.setCurrentIndex(0)
    controls.preview_button.click()

    assert case.view.current_extended_patch is not None
    assert not case.view.current_extended_patch.bezier_mode
    case.view.keyPressEvent(
        type("KeyEvent", (), {"key": lambda self: Qt.Key_B})()
    )

    assert case.view.current_extended_patch.bezier_mode
    assert controls.outer_boundary_combo.currentData() is True
    assert not controls.outer_boundary_combo.isEnabled()
    assert controls.status_message == "Bézier enabled"
    assert case.app is not None


def test_extended_patch_panel_two_cap_preview_and_cancel(monkeypatch):
    case = build_two_cap_selection(monkeypatch, 2, main_edge_count=3)
    controls = grid_editor5.extended_patch_controls(case.view)

    assert "7 boundary edges" in controls.selection_label.text()
    controls.main_direction_combo.setCurrentIndex(1)
    controls.outer_boundary_combo.setCurrentIndex(1)
    controls.preview_button.click()

    patch = case.view.current_extended_patch
    assert patch is not None
    assert patch.bezier_mode
    assert not controls.main_direction_combo.isEnabled()
    assert controls.main_direction_combo.currentData() == 1
    assert controls.radial_layers_spin.value() == 2
    assert not controls.radial_layers_spin.isEnabled()
    assert controls.commit_button.isEnabled()
    assert controls.status_message == "Bézier preview created"

    controls.cancel_button.click()
    assert case.view.current_extended_patch is None
    assert case.view.selected_edges == []
    assert controls.main_direction_combo.isEnabled()
    assert controls.main_direction_combo.currentData() is None
    assert controls.outer_boundary_combo.currentData() is True
    assert controls.status_message == "Ready"
    assert case.app is not None


def test_extended_patch_panel_commit_matches_keyboard_commit(monkeypatch):
    gui_case = build_two_cap_selection(monkeypatch, 1, main_edge_count=1)
    gui_initial_node_count = len(grid_editor5.node_list)
    gui_controls = grid_editor5.extended_patch_controls(gui_case.view)
    gui_controls.main_direction_combo.setCurrentIndex(1)
    gui_controls.preview_button.click()
    gui_controls.commit_button.click()

    assert gui_case.view.current_extended_patch is None
    assert gui_controls.status_message == "Patch committed"
    assert not gui_controls.commit_button.isEnabled()
    gui_result = (
        len(grid_editor5.node_list) - gui_initial_node_count,
        len(grid_editor5.element_list), len(grid_editor5.boundary_list),
    )

    keyboard_case = build_two_cap_selection(
        monkeypatch, 1, main_edge_count=1
    )
    keyboard_initial_node_count = len(grid_editor5.node_list)
    keyboard_case.view.keyPressEvent(
        type("KeyEvent", (), {"key": lambda self: Qt.Key_1})()
    )
    keyboard_case.view.keyPressEvent(
        type("KeyEvent", (), {"key": lambda self: Qt.Key_E})()
    )
    keyboard_case.view.keyPressEvent(
        type("KeyEvent", (), {"key": lambda self: Qt.Key_P})()
    )
    keyboard_result = (
        len(grid_editor5.node_list) - keyboard_initial_node_count,
        len(grid_editor5.element_list), len(grid_editor5.boundary_list),
    )
    assert keyboard_case.view.current_extended_patch is None
    assert keyboard_result == gui_result
    assert gui_case.app is not None


def test_two_cap_preview_reuses_multi_edge_cap_nodes(monkeypatch):
    case = build_two_cap_selection(monkeypatch, 2, main_edge_count=3)

    patch = create_two_cap_patch(case)

    assert patch is not None
    assert patch.radial_layers == 2
    assert len(patch.preview_node_rows) == 3
    for radial_index, row in enumerate(patch.preview_node_rows):
        assert row[0] is case.start_nodes[radial_index]
        assert row[-1] is case.end_nodes[radial_index]
    assert case.app is not None


def test_two_cap_preview_rejects_unequal_cap_lengths_atomically(
    monkeypatch, capsys
):
    case = build_two_cap_selection(
        monkeypatch, 2, main_edge_count=3, end_radial_layers=3
    )
    node_snapshot = list(grid_editor5.node_list)
    element_snapshot = list(grid_editor5.element_list)
    boundary_snapshot = list(grid_editor5.boundary_list)

    patch = create_two_cap_patch(case)

    assert patch is None
    assert grid_editor5.node_list == node_snapshot
    assert grid_editor5.element_list == element_snapshot
    assert grid_editor5.boundary_list == boundary_snapshot
    assert all(edge.active for edge in boundary_snapshot)
    assert (
        "Two-cap extended patch requires equal cap-chain lengths"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "radial_layers, main_edge_count",
    [(1, 3), (2, 3), (4, 3), (2, 1)],
)
def test_multi_layer_two_cap_commit_reuses_both_cap_chains(
    monkeypatch, radial_layers, main_edge_count
):
    case = build_two_cap_selection(
        monkeypatch, radial_layers, main_edge_count=main_edge_count
    )
    patch = create_two_cap_patch(case)
    original_node_count = len(case.nodes)
    start_vectors = [np.array(node.xx, copy=True) for node in case.start_nodes]
    end_vectors = [np.array(node.xx, copy=True) for node in case.end_nodes]

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(patch)

    assert len(elements) == main_edge_count * radial_layers
    assert len(grid_editor5.node_list) == (
        original_node_count + (main_edge_count - 1) * radial_layers
    )
    assert len(node_rows) == radial_layers + 1
    assert all(len(row) == main_edge_count + 1 for row in node_rows)
    for radial_index, row in enumerate(node_rows):
        assert row[0] is case.start_nodes[radial_index]
        assert row[-1] is case.end_nodes[radial_index]

    created_nodes = grid_editor5.node_list[original_node_count:]
    cap_nodes = case.start_nodes + case.end_nodes
    assert all(
        created_node is not cap_node
        and created_node.position != cap_node.position
        for created_node in created_nodes
        for cap_node in cap_nodes
    )
    for radial_index in range(radial_layers):
        layer_elements = elements[
            radial_index * main_edge_count:(radial_index + 1) * main_edge_count
        ]
        assert (
            case.start_cap_edges[radial_index].element_index
            == layer_elements[0].index
        )
        assert (
            case.end_cap_edges[radial_index].element_index
            == layer_elements[-1].index
        )
        for column, element in enumerate(layer_elements):
            assert set(element.vertices) == {
                node_rows[radial_index][column].index,
                node_rows[radial_index][column + 1].index,
                node_rows[radial_index + 1][column].index,
                node_rows[radial_index + 1][column + 1].index,
            }

    consumed_edges = (
        case.main_edges + case.start_cap_edges + case.end_cap_edges
    )
    assert all(edge not in grid_editor5.boundary_list for edge in consumed_edges)
    assert all(not edge.active for edge in consumed_edges)
    assert all(edge.scene() is case.scene for edge in consumed_edges)
    final_boundary_indices = {
        frozenset((node_rows[-1][column].index, node_rows[-1][column + 1].index))
        for column in range(main_edge_count)
    }
    boundary_indices = {
        frozenset(edge.vertices) for edge in grid_editor5.boundary_list
    }
    assert boundary_indices == final_boundary_indices
    for radial_index in range(1, radial_layers):
        intermediate_indices = {
            frozenset((
                node_rows[radial_index][column].index,
                node_rows[radial_index][column + 1].index,
            ))
            for column in range(main_edge_count)
        }
        assert intermediate_indices.isdisjoint(boundary_indices)

    for node, original_xx in zip(case.start_nodes, start_vectors):
        assert np.array_equal(node.xx, original_xx)
    for node, original_xx in zip(case.end_nodes, end_vectors):
        assert np.array_equal(node.xx, original_xx)
    for node in created_nodes:
        assert_basis_handles_match_vectors(node)
    if radial_layers > 1:
        for row in node_rows[1:]:
            for column in range(1, len(row) - 1):
                assert np.dot(
                    row[column].xx[:, 2], node_rows[0][column].xx[:, 2]
                ) > 0.0
    assert_boundary_status_matches_edges(
        grid_editor5.node_list, grid_editor5.boundary_list
    )
    assert all(
        element in node.connected_elements
        for element in elements
        for node in (grid_editor5.node_list[index] for index in element.vertices)
    )
    assert case.view.current_extended_patch is None
    assert patch.scene() is None
    assert case.app is not None


@pytest.mark.parametrize("failure", ["node_count", "edge", "preview"])
def test_multi_layer_two_cap_validation_is_atomic(
    monkeypatch, capsys, failure
):
    case = build_two_cap_selection(monkeypatch, 2, main_edge_count=3)
    patch = create_two_cap_patch(case)
    if failure == "node_count":
        patch.capped_gap.start_cap_nodes.pop()
    elif failure == "edge":
        patch.capped_gap.start_cap_edges[0] = patch.capped_gap.start_cap_edges[1]
    else:
        patch.preview_node_rows = patch.preview_node_rows[:-1]
    node_snapshot = list(grid_editor5.node_list)
    element_snapshot = list(grid_editor5.element_list)
    boundary_snapshot = list(grid_editor5.boundary_list)
    active_snapshot = [edge.active for edge in boundary_snapshot]

    result = grid_editor5.add_extended_patch_to_nodes_elements(patch)

    assert result is None
    assert grid_editor5.node_list == node_snapshot
    assert grid_editor5.element_list == element_snapshot
    assert grid_editor5.boundary_list == boundary_snapshot
    assert [edge.active for edge in boundary_snapshot] == active_snapshot
    assert "Two-cap" in capsys.readouterr().out


@pytest.mark.parametrize(
    "radial_layers, main_edge_count, main_uv_index",
    [(1, 3, 1), (2, 3, 1), (4, 3, 2), (2, 1, 2)],
)
def test_multi_layer_two_cap_bezier_commit_preserves_final_curve(
    monkeypatch, capsys, radial_layers, main_edge_count, main_uv_index
):
    case = build_two_cap_selection(
        monkeypatch, radial_layers, main_edge_count=main_edge_count,
        main_uv_index=main_uv_index,
    )
    patch = create_two_cap_patch(case)
    fixed_start = case.start_nodes[-1]
    fixed_end = case.end_nodes[-1]
    fixed_start.xx[:, main_uv_index] = [0.45, 0.25]
    fixed_end.xx[:, main_uv_index] = [0.4, -0.2]
    grid_editor5.jorek.nodes_xx[:, main_uv_index, fixed_start.index] = (
        fixed_start.xx[:, main_uv_index]
    )
    grid_editor5.jorek.nodes_xx[:, main_uv_index, fixed_end.index] = (
        fixed_end.xx[:, main_uv_index]
    )
    (
        fixed_start.blue_handle if main_uv_index == 1
        else fixed_start.red_handle
    ).sync_position()
    (
        fixed_end.blue_handle if main_uv_index == 1
        else fixed_end.red_handle
    ).sync_position()
    patch.enable_bezier_mode()

    preview_positions = [
        grid_editor5.np_point(node.position) for node in patch.outer_nodes
    ]
    tangents = [np.array(tangent, copy=True) for tangent in patch.outer_tangents]
    parameters = np.array(patch.outer_parameters, copy=True)
    scales = grid_editor5.bezier_nodal_parameter_scales(parameters)
    cap_snapshots = [
        (
            node, grid_editor5.np_point(node.position),
            np.array(node.xx, copy=True),
        )
        for node in case.start_nodes + case.end_nodes
    ]
    original_node_count = len(grid_editor5.node_list)

    commit_result = {}
    real_commit = grid_editor5.add_extended_patch_to_nodes_elements

    def capture_commit(committed_patch):
        commit_result["value"] = real_commit(committed_patch)
        return commit_result["value"]

    monkeypatch.setattr(
        grid_editor5, "add_extended_patch_to_nodes_elements", capture_commit
    )
    case.view.keyPressEvent(
        type("KeyEvent", (), {"key": lambda self: Qt.Key_P})()
    )
    node_rows, elements = commit_result["value"]
    capsys.readouterr()

    assert len(elements) == main_edge_count * radial_layers
    assert len(grid_editor5.node_list) == (
        original_node_count + (main_edge_count - 1) * radial_layers
    )
    assert len(node_rows) == radial_layers + 1
    for radial_index, row in enumerate(node_rows):
        assert row[0] is case.start_nodes[radial_index]
        assert row[-1] is case.end_nodes[radial_index]
        assert all(
            np.allclose(
                node.xx[:, main_uv_index],
                patch.preview_along_vectors[radial_index][column],
            )
            for column, node in enumerate(row)
        )
    for node, position, xx in cap_snapshots:
        assert np.array_equal(grid_editor5.np_point(node.position), position)
        assert np.array_equal(node.xx, xx)
        assert_basis_handles_match_vectors(node)
    assert all(
        np.allclose(grid_editor5.np_point(node.position), preview_position)
        for node, preview_position in zip(node_rows[-1], preview_positions)
    )
    assert all(
        sum(
            np.allclose(
                grid_editor5.np_point(candidate.position),
                grid_editor5.np_point(cap_node.position),
            )
            for candidate in grid_editor5.node_list
        ) == 1
        for cap_node in case.start_nodes + case.end_nodes
    )

    for index, node in enumerate(node_rows[-1][1:-1], start=1):
        assert np.allclose(
            node.xx[:, main_uv_index], scales[index] * tangents[index] / 3.0
        )
        assert np.dot(node.xx[:, main_uv_index], tangents[index]) > 0.0
        assert_basis_handles_match_vectors(node)

    created_edges = {edge for element in elements for edge in element.edges}
    for row_index in range(1, radial_layers + 1):
        for index in range(main_edge_count):
            parameter_start_node = node_rows[row_index][index]
            parameter_end_node = node_rows[row_index][index + 1]
            edge = next(
                edge for edge in created_edges
                if frozenset(edge.vertices) == frozenset((
                    parameter_start_node.index, parameter_end_node.index,
                ))
            )
            if row_index < radial_layers:
                assert np.allclose(np.abs(edge.sizes[1, :]), 1.0)
                continue

            dt = parameters[index + 1] - parameters[index]
            intended_by_node = {
                parameter_start_node: dt * tangents[index] / 3.0,
                parameter_end_node: -dt * tangents[index + 1] / 3.0,
            }
            if parameter_start_node is node_rows[row_index][0]:
                intended_by_node[parameter_start_node] = (
                    patch.preview_along_vectors[row_index][0]
                )
            if parameter_end_node is node_rows[row_index][-1]:
                intended_by_node[parameter_end_node] = -np.asarray(
                    patch.preview_along_vectors[row_index][-1]
                )
            for endpoint, node in enumerate(edge.nodes):
                assert np.allclose(
                    node.xx[:, main_uv_index] * edge.sizes[1, endpoint],
                    intended_by_node[node], atol=1.e-10,
                )

            controls = edge_bezier_points(edge.points)
            controls = np.column_stack((
                controls[:, 0, 0], controls[:, 1, 0],
                controls[:, 1, 1], controls[:, 0, 1],
            ))
            if edge.nodes[0] is not parameter_start_node:
                controls = controls[:, ::-1]
            expected_controls = np.column_stack((
                preview_positions[index],
                preview_positions[index]
                + intended_by_node[parameter_start_node],
                preview_positions[index + 1]
                + intended_by_node[parameter_end_node],
                preview_positions[index + 1],
            ))
            assert np.allclose(controls, expected_controls, atol=1.e-10)
            if index == 0:
                assert np.dot(
                    3.0 * (controls[:, 1] - controls[:, 0]),
                    patch.preview_along_vectors[row_index][0],
                ) > 0.0
            if index == main_edge_count - 1:
                assert np.dot(
                    3.0 * (controls[:, 3] - controls[:, 2]),
                    patch.preview_along_vectors[row_index][-1],
                ) > 0.0

    assert case.view.current_extended_patch is None
    assert patch.scene() is None


def test_two_cap_bezier_fixed_endpoint_validation_is_atomic(
    monkeypatch, capsys
):
    case = build_two_cap_selection(monkeypatch, 2, main_edge_count=3)
    patch = create_two_cap_patch(case)
    patch.enable_bezier_mode()
    case.start_nodes[-1].xx[:, patch.main_uv_index()] = 0.0
    grid_editor5.jorek.nodes_xx[
        :, patch.main_uv_index(), case.start_nodes[-1].index
    ] = 0.0
    node_snapshot = list(grid_editor5.node_list)
    element_snapshot = list(grid_editor5.element_list)
    boundary_snapshot = list(grid_editor5.boundary_list)

    result = grid_editor5.add_extended_patch_to_nodes_elements(patch)

    assert result is None
    assert grid_editor5.node_list == node_snapshot
    assert grid_editor5.element_list == element_snapshot
    assert grid_editor5.boundary_list == boundary_snapshot
    assert "must be nonzero" in capsys.readouterr().out


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_one_cap_gap_creation_normalizes_cap_first_orientation(
    monkeypatch, cap_at_start
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    cap_position = (0.0, 1.0) if cap_at_start else (2.0, 1.0)
    positions = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), cap_position]
    nodes_xx = np.zeros((2, 4, 4))
    for index, position in enumerate(positions):
        nodes_xx[:,0,index] = position
        nodes_xx[:,1,index] = [1.0 / 3.0, 0.0]
        nodes_xx[:,2,index] = [0.0, 1.0 / 3.0]
    nodes_xx[:,2,3] = [0.0, -1.0 / 3.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:,:,index], 2)
        for index in range(4)
    ]
    for node in nodes:
        scene.addItem(node)

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)

    def edge(start, end, uv_index, element_index):
        edge_nodes = [nodes[start], nodes[end]]
        result = grid_editor5.boundary_edge(
            edge_nodes, [start, end], [0,1], element_index, 0, uv_index,
            grid_editor5.signed_edge_sizes(edge_nodes, uv_index),
        )
        scene.addItem(result)
        return result

    inner0 = edge(0, 1, 1, 10)
    inner1 = edge(1, 2, 1, 11)
    cap = edge(0, 3, 2, 12) if cap_at_start else edge(2, 3, 2, 12)
    selected_edges = [inner1, cap, inner0]
    install_edge_owner_lookup(monkeypatch, selected_edges)
    monkeypatch.setattr(
        grid_editor5, "boundary_list", list(selected_edges), raising=False
    )
    topology = grid_editor5.ordered_extended_boundary_topology(
        selected_edges, main_uv_index=1
    )
    patch = grid_editor5.extended_patch(
        topology.inner_nodes, topology.inner_edges, can_commit=True
    )
    patch.one_cap_topology = topology
    patch.set_outer_positions([
        QPointF(0.0, 1.0), QPointF(1.0, 1.0), QPointF(2.0, 1.0)
    ])
    scene.addItem(patch)
    view.current_extended_patch = patch
    view.selected_edges = list(selected_edges)
    old_node_count = len(nodes)

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(patch)

    assert len(grid_editor5.node_list) == old_node_count + 2
    assert len(elements) == 2
    assert node_rows[1][0] is nodes[3]
    assert all(edge not in grid_editor5.boundary_list for edge in selected_edges)
    assert len(grid_editor5.boundary_list) == 3

    def position(index):
        point = grid_editor5.node_list[index].position
        return (point.x(), point.y())

    boundary_geometry = {
        frozenset(position(vertex) for vertex in edge.vertices)
        for edge in grid_editor5.boundary_list
    }
    assert boundary_geometry == {
        frozenset(((0.0, 1.0), (1.0, 1.0))),
        frozenset(((1.0, 1.0), (2.0, 1.0))),
        frozenset(
            ((2.0, 0.0), (2.0, 1.0))
            if cap_at_start else ((0.0, 0.0), (0.0, 1.0))
        ),
    }
    element_geometry = {
        frozenset(position(vertex) for vertex in element.vertices)
        for element in elements
    }
    assert element_geometry == {
        frozenset(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        frozenset(((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0))),
    }
    assert view.current_extended_patch is None
    assert app is not None


def build_multi_layer_one_cap_case(
    monkeypatch, radial_layers, cap_at_start=True, main_edge_count=2,
    main_uv_index=1,
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    main_node_count = main_edge_count + 1
    radial_uv_index = main_uv_index % 2 + 1
    cap_x = 0.0 if cap_at_start else float(main_edge_count)
    positions = [(float(index), 0.0) for index in range(main_node_count)]
    positions.extend(
        (cap_x, float(radial_index))
        for radial_index in range(1, radial_layers + 1)
    )
    nodes_xx = np.zeros((2, 4, len(positions)))
    for index, position in enumerate(positions):
        nodes_xx[:, 0, index] = position
        nodes_xx[:, main_uv_index, index] = [1.0 / 3.0, 0.0]
        nodes_xx[:, radial_uv_index, index] = [0.0, 1.0 / 3.0]
    nodes = [
        jorek_node_item(index, nodes_xx[:, :, index], 2)
        for index in range(len(positions))
    ]
    for node in nodes:
        scene.addItem(node)

    monkeypatch.setattr(grid_editor5, "jorek", SimpleNamespace(nodes_xx=nodes_xx))
    monkeypatch.setattr(grid_editor5, "this_scaling", 1.0, raising=False)
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [], raising=False)
    monkeypatch.setattr(grid_editor5, "scene", scene, raising=False)
    monkeypatch.setattr(grid_editor5, "view", view, raising=False)

    def edge(start, end, uv_index, element_index):
        edge_nodes = [nodes[start], nodes[end]]
        result = grid_editor5.boundary_edge(
            edge_nodes, [start, end], [0, 1], element_index, 0, uv_index,
            grid_editor5.signed_edge_sizes(edge_nodes, uv_index),
        )
        scene.addItem(result)
        return result

    main_edges = [
        edge(index, index + 1, main_uv_index, 10 + index)
        for index in range(main_edge_count)
    ]
    cap_edges = []
    previous = 0 if cap_at_start else main_edge_count
    for radial_index in range(radial_layers):
        cap_node_index = main_node_count + radial_index
        cap_edges.append(edge(
            previous, cap_node_index, radial_uv_index, 20 + radial_index
        ))
        previous = cap_node_index
    selected_edges = list(reversed(main_edges + cap_edges))
    owners = install_edge_owner_lookup(monkeypatch, selected_edges)
    monkeypatch.setattr(
        grid_editor5, "boundary_list", list(selected_edges), raising=False
    )
    topology = grid_editor5.ordered_extended_boundary_topology(
        selected_edges, main_uv_index=main_uv_index
    )
    patch = grid_editor5.extended_patch(
        topology.inner_nodes, topology.inner_edges, can_commit=True
    )
    patch.one_cap_topology = topology
    patch.radial_layers = radial_layers
    patch.set_outer_positions([
        QPointF(float(column), float(radial_layers))
        for column in range(main_node_count)
    ])
    scene.addItem(patch)
    view.current_extended_patch = patch
    view.selected_edges = list(selected_edges)
    cap_nodes = (
        topology.start_cap_nodes if cap_at_start else topology.end_cap_nodes
    )
    return SimpleNamespace(
        app=app, scene=scene, view=view, nodes=nodes, patch=patch,
        main_edges=main_edges, cap_edges=cap_edges,
        cap_nodes=cap_nodes, selected_edges=selected_edges,
        owners=owners,
    )


def attach_old_cap_neighbors(case, cap_edges, cap_nodes, old_lengths, direction):
    """Complete lightweight cap owners with their old-side logical neighbor."""
    main_uv_index = case.patch.main_uv_index()
    for cap_edge, cap_node, old_length in zip(
        cap_edges, cap_nodes[1:], old_lengths
    ):
        owner = case.owners[cap_edge.element_index]
        local_vertex = list(owner.vertices).index(cap_node.index)
        adjacent_sides = {(local_vertex - 1) % 4, local_vertex}
        old_side = (adjacent_sides - {cap_edge.element_side}).pop()
        old_local_vertex = (
            old_side if (old_side + 1) % 4 == local_vertex
            else (local_vertex + 1) % 4
        )
        position = (
            grid_editor5.np_point(cap_node.position)
            - old_length * np.asarray(direction, dtype=float)
        )
        node_index = len(grid_editor5.node_list)
        xx = np.zeros((2, 4))
        xx[:, 0] = position
        xx[:, main_uv_index] = [1.0 / 3.0, 0.0]
        xx[:, main_uv_index % 2 + 1] = [0.0, 1.0 / 3.0]
        old_node = jorek_node_item(node_index, xx, 2)
        grid_editor5.node_list.append(old_node)
        case.scene.addItem(old_node)
        grid_editor5.jorek.nodes_xx = np.append(
            grid_editor5.jorek.nodes_xx, xx[:, :, None], axis=2
        )
        owner.vertices[old_local_vertex] = node_index


@pytest.mark.parametrize("new_length", [0.5, 1.0, 2.0])
def test_cap_continuation_vector_scales_with_local_chord(monkeypatch, new_length):
    cap = SimpleNamespace(
        index=1, position=QPointF(0.0, 0.0),
        xx=np.array([[0.0, 0.25, 0.0, 0.0],
                     [0.0, 0.0, 0.25, 0.0]]),
    )
    nodes = [
        SimpleNamespace(index=0, position=QPointF(0.0, -1.0)),
        cap,
        SimpleNamespace(index=2, position=QPointF(-1.0, 0.0)),
        SimpleNamespace(index=3, position=QPointF(-1.0, -1.0)),
    ]
    owner = SimpleNamespace(
        index=7, vertices=np.array([0, 1, 2, 3]),
        sizes=np.ones((4, 4)), active=True,
    )
    owner.sizes[1, 1] = -2.0
    cap_edge = SimpleNamespace(
        element_index=7, element_side=0,
        nodes=[nodes[0], cap], sizes=np.ones((2, 2)),
    )
    monkeypatch.setattr(grid_editor5, "node_list", nodes, raising=False)
    monkeypatch.setattr(grid_editor5, "element_list", [owner], raising=False)

    old_effective = -2.0 * cap.xx[:, 1]
    result = grid_editor5.cap_continuation_basis_vector(
        cap, QPointF(new_length, 0.0), 1, cap_edge
    )

    assert np.isclose(
        np.linalg.norm(result) / np.linalg.norm(old_effective), new_length
    )
    assert np.isclose(
        np.dot(result, -old_effective)
        / (np.linalg.norm(result) * np.linalg.norm(old_effective)),
        1.0,
    )
    assert np.linalg.norm(result) > 1.e-12


def test_multi_layer_one_cap_scaled_vectors_survive_commit(monkeypatch):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=3, cap_at_start=True, main_edge_count=2
    )
    old_lengths = [0.5, 1.0, 2.0]
    attach_old_cap_neighbors(
        case, case.cap_edges, case.cap_nodes, old_lengths, [1.0, 0.0]
    )
    cap_snapshots = [np.array(node.xx, copy=True) for node in case.cap_nodes]
    case.patch.enable_bezier_mode()

    desired = {}
    for radial_index, old_length in enumerate(old_lengths, start=1):
        row = case.patch.preview_node_rows[radial_index]
        new_length = np.linalg.norm(
            grid_editor5.np_point(row[1].position)
            - grid_editor5.np_point(row[0].position)
        )
        old_vector = np.array([1.0 / 3.0, 0.0])
        vector = np.asarray(
            case.patch.preview_along_vectors[radial_index][0]
        )
        assert np.isclose(
            np.linalg.norm(vector) / np.linalg.norm(old_vector),
            new_length / old_length,
        )
        assert np.isclose(
            np.dot(vector, old_vector)
            / (np.linalg.norm(vector) * np.linalg.norm(old_vector)), 1.0
        )
        desired[case.cap_nodes[radial_index].index] = vector

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(
        case.patch
    )
    assert all(
        np.array_equal(node.xx, snapshot)
        for node, snapshot in zip(case.cap_nodes, cap_snapshots)
    )
    created_edges = {edge for element in elements for edge in element.edges}
    for radial_index in range(1, 4):
        cap_node, next_node = node_rows[radial_index][:2]
        edge = next(
            edge for edge in created_edges
            if frozenset(edge.vertices)
            == frozenset((cap_node.index, next_node.index))
        )
        endpoint = next(
            index for index, node in enumerate(edge.nodes) if node is cap_node
        )
        assert np.allclose(
            cap_node.xx[:, case.patch.main_uv_index()]
            * edge.sizes[1, endpoint],
            desired[cap_node.index],
        )


def test_two_cap_preview_scales_both_cap_continuations(monkeypatch):
    case = build_two_cap_selection(monkeypatch, 2, main_edge_count=3)
    patch = create_two_cap_patch(case)
    case.patch = patch
    attach_old_cap_neighbors(
        case, case.start_cap_edges, case.start_nodes, [0.5, 2.0], [1.0, 0.0]
    )
    attach_old_cap_neighbors(
        case, case.end_cap_edges, case.end_nodes, [2.0, 0.5], [-1.0, 0.0]
    )
    patch.enable_bezier_mode()

    for radial_index in (1, 2):
        row = patch.preview_node_rows[radial_index]
        for column, old_length, direction in (
            (0, [0.5, 2.0][radial_index - 1], np.array([1.0, 0.0])),
            (-1, [2.0, 0.5][radial_index - 1], np.array([-1.0, 0.0])),
        ):
            neighbor = row[1] if column == 0 else row[-2]
            new_length = np.linalg.norm(
                grid_editor5.np_point(neighbor.position)
                - grid_editor5.np_point(row[column].position)
            )
            vector = np.asarray(
                patch.preview_along_vectors[radial_index][column]
            )
            assert np.isclose(np.linalg.norm(vector), new_length / old_length / 3.0)
            oriented_vector = vector if column == 0 else -vector
            assert np.dot(oriented_vector, direction) > 0.0


@pytest.mark.parametrize("cap_at_start", [True, False])
@pytest.mark.parametrize("requested_new_length", [1.0, 2.0])
def test_one_cap_terminal_bezier_endpoint_uses_scaled_continuation(
    monkeypatch, cap_at_start, requested_new_length
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=3,
    )
    continuation = np.array([1.0, 0.0]) if cap_at_start else np.array([-1.0, 0.0])
    attach_old_cap_neighbors(
        case, case.cap_edges, case.cap_nodes, [1.0, 1.0], continuation
    )
    fixed_node = case.cap_nodes[-1]
    fixed_position = grid_editor5.np_point(fixed_node.position)
    fixed_xx = np.array(fixed_node.xx, copy=True)
    fixed_x = fixed_position[0]
    if cap_at_start:
        outer_positions = [
            QPointF(fixed_x + requested_new_length * index, fixed_position[1])
            for index in range(4)
        ]
    else:
        outer_positions = [
            QPointF(
                fixed_x - requested_new_length * (3 - index),
                fixed_position[1],
            )
            for index in range(4)
        ]
    case.patch.set_outer_positions(outer_positions)
    case.patch.enable_bezier_mode()

    endpoint_vector = (
        case.patch.bezier_start_vector()
        if cap_at_start else case.patch.bezier_end_vector()
    )
    neighbor = (
        case.patch.outer_nodes[1]
        if cap_at_start else case.patch.outer_nodes[-2]
    )
    new_length = np.linalg.norm(
        grid_editor5.np_point(neighbor.position) - fixed_position
    )
    old_effective = grid_editor5.effective_node_basis_vector(
        fixed_node, case.patch.main_uv_index(), case.cap_edges[-1]
    )
    if np.dot(old_effective, continuation) < 0.0:
        old_effective = -old_effective

    assert case.patch.one_cap_global_active
    global_data = case.patch.one_cap_global_rows[case.patch.radial_layers]
    endpoint_span = (
        global_data["parameters"][1]
        if cap_at_start
        else 1.0 - global_data["parameters"][-2]
    )
    recovered_local = endpoint_span * endpoint_vector
    assert np.isclose(
        np.linalg.norm(recovered_local) / np.linalg.norm(old_effective),
        requested_new_length,
    )
    assert np.isclose(
        np.dot(recovered_local, old_effective)
        / (np.linalg.norm(recovered_local) * np.linalg.norm(old_effective)),
        1.0,
    )
    assert np.linalg.norm(recovered_local) > 1.e-12
    assert np.array_equal(grid_editor5.np_point(fixed_node.position), fixed_position)
    assert np.array_equal(fixed_node.xx, fixed_xx)

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(
        case.patch
    )
    # One-cap commit normalizes rows into cap-first topology order.
    cap_column = 0
    neighbor_column = 1
    cap_node = node_rows[-1][cap_column]
    new_neighbor = node_rows[-1][neighbor_column]
    outer_edge = next(
        edge for element in elements for edge in element.edges
        if frozenset(edge.vertices)
        == frozenset((cap_node.index, new_neighbor.index))
    )
    endpoint = next(
        index for index, node in enumerate(outer_edge.nodes)
        if node is cap_node
    )
    committed_effective = (
        cap_node.xx[:, case.patch.main_uv_index()]
        * outer_edge.sizes[1, endpoint]
    )
    assert np.allclose(committed_effective, recovered_local)
    assert np.array_equal(grid_editor5.np_point(fixed_node.position), fixed_position)
    assert np.array_equal(fixed_node.xx, fixed_xx)


def test_graphics_items_use_explicit_visual_layering(monkeypatch):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=1, cap_at_start=True
    )
    case.patch.enable_bezier_mode()
    case.view.selected_edges = [case.main_edges[0]]
    single_patch_preview = big_patch()

    assert (
        grid_editor5.Z_STATIC_MESH
        < grid_editor5.Z_MESH_EDGE
        < grid_editor5.Z_BOUNDARY_EDGE
        < grid_editor5.Z_PATCH_PREVIEW
        < grid_editor5.Z_NODE
        < grid_editor5.Z_VECTOR_HANDLE
    )
    assert all(
        edge.zValue() == grid_editor5.Z_BOUNDARY_EDGE
        for edge in case.main_edges + case.cap_edges
    )
    assert case.patch.zValue() == grid_editor5.Z_PATCH_PREVIEW
    assert single_patch_preview.zValue() == grid_editor5.Z_PATCH_PREVIEW
    assert all(node.zValue() == grid_editor5.Z_NODE for node in case.nodes)
    assert all(
        handle.parentItem() is node
        and handle.zValue() == grid_editor5.Z_VECTOR_HANDLE
        for node in case.nodes
        for handle in (node.blue_handle, node.red_handle)
    )
    assert all(
        handle.parentItem() is case.patch
        and handle.zValue() == grid_editor5.Z_VECTOR_HANDLE
        for handle in case.patch.bezier_handles
    )

    unused_rows, elements = (
        grid_editor5.add_extended_patch_to_nodes_elements(case.patch)
    )
    assert elements
    assert all(
        element.zValue() == grid_editor5.Z_MESH_EDGE
        for element in elements
    )
    assert case.app is not None


@pytest.mark.parametrize(
    "radial_layers, cap_at_start",
    [(1, True), (2, True), (4, True), (2, False)],
)
def test_multi_layer_one_cap_commit_reuses_cap_chain_nodes(
    monkeypatch, radial_layers, cap_at_start
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers, cap_at_start
    )
    original_node_count = len(case.nodes)
    main_edge_count = len(case.main_edges)

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(
        case.patch
    )

    assert len(elements) == main_edge_count * radial_layers
    assert len(grid_editor5.node_list) == (
        original_node_count + main_edge_count * radial_layers
    )
    assert len(node_rows) == radial_layers + 1
    assert all(len(row) == main_edge_count + 1 for row in node_rows)
    for radial_index, row in enumerate(node_rows):
        assert row[0] is case.cap_nodes[radial_index]
    created_nodes = grid_editor5.node_list[original_node_count:]
    assert all(
        created_node is not cap_node
        and created_node.position != cap_node.position
        for created_node in created_nodes
        for cap_node in case.cap_nodes
    )

    for radial_index in range(radial_layers):
        layer_elements = elements[
            radial_index * main_edge_count:(radial_index + 1) * main_edge_count
        ]
        assert case.cap_edges[radial_index].element_index == layer_elements[0].index
        for column, element in enumerate(layer_elements):
            assert set(element.vertices) == {
                node_rows[radial_index][column].index,
                node_rows[radial_index][column + 1].index,
                node_rows[radial_index + 1][column].index,
                node_rows[radial_index + 1][column + 1].index,
            }

    for consumed_edge in case.main_edges + case.cap_edges:
        assert consumed_edge not in grid_editor5.boundary_list
        assert not consumed_edge.active
        assert consumed_edge.scene() is case.scene
    final_row_edge_indices = {
        frozenset((node_rows[-1][column].index, node_rows[-1][column + 1].index))
        for column in range(main_edge_count)
    }
    boundary_indices = {
        frozenset(edge.vertices) for edge in grid_editor5.boundary_list
    }
    assert final_row_edge_indices.issubset(boundary_indices)
    uncapped_side_indices = {
        frozenset((node_rows[row][-1].index, node_rows[row + 1][-1].index))
        for row in range(radial_layers)
    }
    assert uncapped_side_indices.issubset(boundary_indices)
    for radial_index in range(1, radial_layers):
        intermediate_indices = {
            frozenset((
                node_rows[radial_index][column].index,
                node_rows[radial_index][column + 1].index,
            ))
            for column in range(main_edge_count)
        }
        assert intermediate_indices.isdisjoint(boundary_indices)

    for node in grid_editor5.node_list:
        assert_basis_handles_match_vectors(node)
    assert_boundary_status_matches_edges(
        grid_editor5.node_list, grid_editor5.boundary_list
    )
    assert case.view.current_extended_patch is None
    assert case.patch.scene() is None
    assert case.app is not None


@pytest.mark.parametrize("failure", ["preview", "cap_count"])
def test_multi_layer_one_cap_validation_fails_before_mesh_mutation(
    monkeypatch, capsys, failure
):
    case = build_multi_layer_one_cap_case(monkeypatch, 2, cap_at_start=True)
    if failure == "preview":
        case.patch.preview_node_rows = case.patch.preview_node_rows[:-1]
    else:
        case.patch.radial_layers = 3
    node_snapshot = list(grid_editor5.node_list)
    element_snapshot = list(grid_editor5.element_list)
    boundary_snapshot = list(grid_editor5.boundary_list)

    result = grid_editor5.add_extended_patch_to_nodes_elements(case.patch)

    assert result is None
    assert grid_editor5.node_list == node_snapshot
    assert grid_editor5.element_list == element_snapshot
    assert grid_editor5.boundary_list == boundary_snapshot
    assert "One-cap" in capsys.readouterr().out


@pytest.mark.parametrize(
    "radial_layers, cap_at_start, main_uv_index",
    [(1, True, 1), (2, True, 1), (4, True, 2), (2, False, 2)],
)
def test_multi_layer_one_cap_bezier_commit_preserves_final_curve(
    monkeypatch, capsys, radial_layers, cap_at_start, main_uv_index
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers, cap_at_start,
        main_edge_count=3, main_uv_index=main_uv_index,
    )
    patch = case.patch
    patch.enable_bezier_mode()
    handles = {handle.role: handle for handle in patch.bezier_handles}
    endpoint_role = "end" if cap_at_start else "start"
    endpoint_handle = handles[endpoint_role]
    endpoint_handle.move_to_scene(
        endpoint_handle.mapToScene(QPointF(0.0, 0.0)) + QPointF(0.15, 0.1)
    )
    tangent_role = "end_tangent" if cap_at_start else "start_tangent"
    tangent_handle = handles[tangent_role]
    tangent_handle.move_to_scene(
        tangent_handle.mapToScene(QPointF(0.0, 0.0)) + QPointF(0.0, 0.3)
    )

    topology = patch.one_cap_topology
    working_tangents, working_parameters = grid_editor5.cap_first_bezier_data(
        patch, topology
    )
    working_positions = [
        grid_editor5.np_point(node.position)
        for node in (
            patch.outer_nodes
            if cap_at_start else list(reversed(patch.outer_nodes))
        )
    ]
    fixed_node = case.cap_nodes[-1]
    fixed_position = grid_editor5.np_point(fixed_node.position)
    fixed_xx = np.array(fixed_node.xx, copy=True)
    original_node_count = len(grid_editor5.node_list)

    commit_result = {}
    real_commit = grid_editor5.add_extended_patch_to_nodes_elements

    def capture_commit(committed_patch):
        commit_result["value"] = real_commit(committed_patch)
        return commit_result["value"]

    monkeypatch.setattr(
        grid_editor5, "add_extended_patch_to_nodes_elements", capture_commit
    )
    case.view.keyPressEvent(type("KeyEvent", (), {"key": lambda self: Qt.Key_P})())
    node_rows, elements = commit_result["value"]
    capsys.readouterr()

    edge_count = len(case.main_edges)
    radial_uv_index = main_uv_index % 2 + 1
    assert len(elements) == edge_count * radial_layers
    assert len(grid_editor5.node_list) == original_node_count + edge_count * radial_layers
    assert len({node.index for row in node_rows for node in row}) == (
        (edge_count + 1) * (radial_layers + 1)
    )
    assert node_rows[-1][0] is fixed_node
    assert np.array_equal(grid_editor5.np_point(fixed_node.position), fixed_position)
    assert np.array_equal(fixed_node.xx, fixed_xx)
    assert_basis_handles_match_vectors(fixed_node)
    assert [
        node for node in grid_editor5.node_list
        if np.allclose(grid_editor5.np_point(node.position), fixed_position)
    ] == [fixed_node]
    assert all(
        np.allclose(grid_editor5.np_point(node.position), expected)
        for node, expected in zip(node_rows[-1], working_positions)
    )

    scales = grid_editor5.bezier_nodal_parameter_scales(working_parameters)
    for row_index, row in enumerate(node_rows[1:-1], start=1):
        row_tangents, row_parameters, unused_segments = (
            grid_editor5.cap_first_global_bezier_row_data(
                patch, topology, row_index
            )
        )
        row_scales = grid_editor5.bezier_nodal_parameter_scales(row_parameters)
        for column, node in enumerate(row[1:], start=1):
            assert np.allclose(
                node.xx[:, main_uv_index],
                row_scales[column] * row_tangents[column] / 3.0,
            )
    for index, node in enumerate(node_rows[-1][1:], start=1):
        assert np.allclose(
            node.xx[:, main_uv_index],
            scales[index] * working_tangents[index] / 3.0,
        )
    for node in node_rows[-1]:
        assert_basis_handles_match_vectors(node)

    created_edges = {edge for element in elements for edge in element.edges}
    for row_index in range(1, radial_layers + 1):
        for index in range(edge_count):
            node0 = node_rows[row_index][index]
            node1 = node_rows[row_index][index + 1]
            edge = next(
                edge for edge in created_edges
                if frozenset(edge.vertices) == frozenset((node0.index, node1.index))
            )
            row_tangents, row_parameters, row_segments = (
                grid_editor5.cap_first_global_bezier_row_data(
                    patch, topology, row_index
                )
            )
            dt = row_parameters[index + 1] - row_parameters[index]
            expected_effective = {
                node0: dt * row_tangents[index] / 3.0,
                node1: -dt * row_tangents[index + 1] / 3.0,
            }
            for endpoint, node in enumerate(edge.nodes):
                assert np.allclose(
                    node.xx[:, main_uv_index] * edge.sizes[1, endpoint],
                    expected_effective[node], atol=1.e-10,
                )
            if row_index == radial_layers and index == edge_count - 1:
                free_node = node_rows[-1][-1]
                free_endpoint = next(
                    endpoint for endpoint, node in enumerate(edge.nodes)
                    if node is free_node
                )
                assert free_node is node1
                assert edge.nodes[free_endpoint] is node1
                assert np.dot(
                    free_node.xx[:, main_uv_index], row_tangents[-1]
                ) > 0.0
                effective_end = (
                    free_node.xx[:, main_uv_index]
                    * edge.sizes[1, free_endpoint]
                )
                intended_end = -dt * row_tangents[-1] / 3.0
                assert np.allclose(effective_end, intended_end, atol=1.e-10)
                endpoint_position = edge.points[:, 0, free_endpoint]
                endpoint_control = (
                    endpoint_position + edge.points[:, 1, free_endpoint]
                )
                derivative_at_end = 3.0 * (
                    endpoint_position - endpoint_control
                )
                assert np.dot(derivative_at_end, row_tangents[-1]) > 0.0
            controls = edge_bezier_points(edge.points)
            controls = np.column_stack((
                controls[:, 0, 0], controls[:, 1, 0],
                controls[:, 1, 1], controls[:, 0, 1],
            ))
            if edge.nodes[0] is not node0:
                controls = controls[:, ::-1]
            assert np.allclose(
                controls, np.asarray(row_segments[index]).T, atol=1.e-10
            )

    assert case.view.current_extended_patch is None
    assert patch.scene() is None


def test_end_cap_bezier_data_reverses_parameters_and_tangents(monkeypatch):
    case = build_multi_layer_one_cap_case(
        monkeypatch, 2, cap_at_start=False, main_edge_count=3
    )
    case.patch.enable_bezier_mode()
    original_parameters = np.array(case.patch.outer_parameters, copy=True)
    original_tangents = [np.array(tangent, copy=True) for tangent in case.patch.outer_tangents]

    tangents, parameters = grid_editor5.cap_first_bezier_data(
        case.patch, case.patch.one_cap_topology
    )

    assert np.allclose(parameters, 1.0 - original_parameters[::-1])
    assert all(
        np.allclose(tangent, -original)
        for tangent, original in zip(tangents, reversed(original_tangents))
    )
    assert np.all(np.diff(parameters) > 0.0)


def cap_global_handle_geometry(patch, cap_at_start):
    handle = patch.global_cap_tangent_handle()
    fixed_position = (
        patch.bezier_start_position()
        if cap_at_start else patch.bezier_end_position()
    )
    return (
        handle,
        grid_editor5.np_point(fixed_position),
        np.asarray(patch.global_cap_tangent_direction, dtype=float),
    )


def drag_cap_global_handle(patch, cap_at_start, length, lateral=0.0):
    handle, fixed, direction = cap_global_handle_geometry(
        patch, cap_at_start
    )
    normal = np.array([-direction[1], direction[0]])
    requested = fixed + length * direction + lateral * normal
    return handle, patch.move_global_cap_tangent_handle(
        handle, patch.mapToScene(QPointF(*requested))
    )


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_one_cap_global_tangent_handle_initial_state_and_constraint(
    monkeypatch, cap_at_start
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=3,
    )
    patch = case.patch
    patch.enable_bezier_mode()
    role = (
        "global_start_tangent"
        if cap_at_start else "global_end_tangent"
    )
    handles = {handle.role: handle for handle in patch.bezier_handles}
    handle, fixed, direction = cap_global_handle_geometry(
        patch, cap_at_start
    )
    data = patch.one_cap_global_rows[patch.radial_layers]
    expected_control = data["controls"][1 if cap_at_start else 2]

    assert handle is handles[role]
    assert (
        "end_tangent" if cap_at_start else "start_tangent"
    ) in handles
    assert np.allclose(grid_editor5.np_point(handle.pos()), expected_control)
    assert not np.allclose(grid_editor5.np_point(handle.pos()), fixed)
    assert handle.isVisible()
    assert handle.isEnabled()
    assert handle.acceptedMouseButtons() & Qt.LeftButton
    assert (
        handle.flags()
        & grid_editor5.QGraphicsItem.ItemIgnoresTransformations
    )
    assert handle.parentItem() is patch
    assert handle.zValue() == grid_editor5.Z_VECTOR_HANDLE
    assert np.isclose(
        patch.global_cap_tangent_length,
        patch.global_cap_tangent_default_length,
    )

    # Removing the explicit state reproduces the pre-handle automatic curve.
    initial_rows = {
        index: np.array(row["controls"], copy=True)
        for index, row in patch.one_cap_global_rows.items()
    }
    initial_positions = [
        [grid_editor5.np_point(node.position) for node in row]
        for row in patch.preview_node_rows
    ]
    initial_length = patch.global_cap_tangent_length
    patch.global_cap_tangent_length = None
    patch.update_bezier_from_handles()
    assert all(
        np.allclose(row["controls"], initial_rows[index], atol=2.e-10)
        for index, row in patch.one_cap_global_rows.items()
    )
    assert all(
        np.allclose(grid_editor5.np_point(node.position), expected)
        for row, expected_row in zip(
            patch.preview_node_rows, initial_positions
        )
        for node, expected in zip(row, expected_row)
    )
    patch.global_cap_tangent_length = initial_length
    patch.update_bezier_from_handles()

    target_length = 1.5 * initial_length
    handle, accepted = drag_cap_global_handle(
        patch, cap_at_start, target_length, lateral=7.0
    )
    actual_vector = grid_editor5.np_point(handle.pos()) - fixed
    assert accepted
    assert np.isclose(np.dot(actual_vector, direction), target_length)
    assert abs(np.cross(direction, actual_vector)) < 2.e-12
    assert np.allclose(
        actual_vector / np.linalg.norm(actual_vector), direction
    )
    assert not patch.automatic_outer_geometry

    # A drag behind the cap clamps to a small positive geometry-scaled value.
    handle, accepted = drag_cap_global_handle(
        patch, cap_at_start, -100.0, lateral=-11.0
    )
    actual_vector = grid_editor5.np_point(handle.pos()) - fixed
    assert accepted
    assert np.dot(actual_vector, direction) > 0.0
    assert abs(np.cross(direction, actual_vector)) < 2.e-12


@pytest.mark.parametrize("cap_at_start", [True, False])
@pytest.mark.parametrize("main_edge_count", [3, 40])
def test_production_preview_path_creates_isolated_one_cap_global_curve(
    monkeypatch, cap_at_start, main_edge_count
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=main_edge_count,
    )
    case.scene.removeItem(case.patch)
    case.view.current_extended_patch = None
    case.view.selected_edges = list(case.selected_edges)
    case.view.pending_main_uv_index = 1
    case.view.pending_bezier_mode = True
    monkeypatch.setattr(
        grid_editor5, "subdivide_cubic_bezier_controls",
        lambda *args, **kwargs: pytest.fail(
            "authoritative outer curve used element subdivision"
        ),
    )
    monkeypatch.setattr(
        grid_editor5, "global_cubic_from_local_endpoint_controls",
        lambda *args, **kwargs: pytest.fail(
            "isolated curve used local/subdivision endpoint controls"
        ),
    )
    monkeypatch.setattr(
        grid_editor5, "validate_bezier_control_nets",
        lambda *args, **kwargs: pytest.fail(
            "isolated curve ran patch Jacobian validation"
        ),
    )

    assert case.view.create_extended_patch_preview()
    patch = case.view.current_extended_patch
    handles = {handle.role: handle for handle in patch.bezier_handles}
    global_handles = [
        handle for handle in patch.bezier_handles
        if handle.role == "cap_global_tangent"
    ]
    handle = global_handles[0]
    fixed = grid_editor5.np_point(
        patch.bezier_start_position()
        if cap_at_start else patch.bezier_end_position()
    )
    expected = (
        fixed
        + patch.global_cap_tangent_length
        * np.asarray(patch.global_cap_tangent_direction)
    )

    expected_roles = (
        {"end", "end_tangent", "cap_global_tangent"}
        if cap_at_start
        else {"start", "start_tangent", "cap_global_tangent"}
    )
    assert patch.one_cap_global_curve_only
    assert {item.role for item in patch.bezier_handles} == expected_roles
    assert len(global_handles) == 1
    assert np.allclose(grid_editor5.np_point(handle.pos()), expected)
    assert not np.allclose(grid_editor5.np_point(handle.pos()), fixed)
    assert handle.isVisible()
    assert handle.isEnabled()
    assert handle.acceptedMouseButtons() & Qt.LeftButton
    assert (
        handle.flags()
        & grid_editor5.QGraphicsItem.ItemIgnoresTransformations
    )
    assert handle.parentItem() is patch
    controls = np.asarray(patch.one_cap_global_curve_controls)
    controls_before_preview_checks = np.array(controls, copy=True)
    cap_control_index = 1 if cap_at_start else 2
    fixed_control_index = 0 if cap_at_start else 3
    assert np.isclose(
        np.linalg.norm(
            controls[cap_control_index] - controls[fixed_control_index]
        ),
        np.linalg.norm(controls[3] - controls[0]) / 3.0,
    )
    assert np.array_equal(qpath_cubic_controls(patch.path()), controls)
    assert np.array_equal(
        qpath_cubic_controls(patch.one_cap_global_control_polygon.path()),
        controls,
    )
    assert len(patch.outer_nodes) == main_edge_count + 1
    assert len(patch.outer_parameters) == main_edge_count + 1
    for node, tangent, parameter in zip(
        patch.outer_nodes, patch.outer_tangents, patch.outer_parameters
    ):
        expected_point, expected_tangent = (
            grid_editor5.cubic_bezier_point_and_tangent_from_controls(
                controls, parameter
            )
        )
        assert np.allclose(grid_editor5.np_point(node.position), expected_point)
        assert np.allclose(tangent, expected_tangent)
    topology = patch.one_cap_topology
    cap_column = 0 if cap_at_start else -1
    cap_nodes = (
        topology.start_cap_nodes
        if cap_at_start else topology.end_cap_nodes
    )
    for radial_index in range(1, patch.radial_layers + 1):
        assert (
            patch.preview_node_rows[radial_index][cap_column]
            is cap_nodes[radial_index]
        )
    assert patch.outer_nodes[cap_column] is cap_nodes[-1]
    free_column = -1 if cap_at_start else 0
    free_role = "end" if cap_at_start else "start"
    assert patch.outer_nodes[free_column].position == handles[free_role].pos()
    assert np.array_equal(
        patch.one_cap_global_curve_controls,
        controls_before_preview_checks,
    )


def qpath_cubic_controls(path):
    assert path.elementCount() >= 4
    return np.array([
        [path.elementAt(index).x, path.elementAt(index).y]
        for index in range(4)
    ])


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_isolated_one_cap_curve_handles_update_only_global_cubic(
    monkeypatch, cap_at_start
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=3,
    )
    case.scene.removeItem(case.patch)
    case.view.current_extended_patch = None
    case.view.selected_edges = list(case.selected_edges)
    case.view.pending_main_uv_index = 1
    case.view.pending_bezier_mode = True
    assert case.view.create_extended_patch_preview()
    patch = case.view.current_extended_patch
    handles = {handle.role: handle for handle in patch.bezier_handles}
    cap_handle = handles["cap_global_tangent"]
    endpoint_role = "end" if cap_at_start else "start"
    tangent_role = "end_tangent" if cap_at_start else "start_tangent"
    endpoint_handle = handles[endpoint_role]
    tangent_handle = handles[tangent_role]
    cap_control_index = 1 if cap_at_start else 2
    endpoint_index = 3 if cap_at_start else 0
    tangent_index = 2 if cap_at_start else 1
    fixed_index = 0 if cap_at_start else 3
    original = np.array(patch.one_cap_global_curve_controls, copy=True)
    original_direction = np.array(
        patch.global_cap_tangent_direction, copy=True
    )
    outer_before_cap = np.array([
        grid_editor5.np_point(node.position) for node in patch.outer_nodes
    ])
    middle_before_cap = np.array([
        grid_editor5.np_point(node.position)
        for node in patch.preview_node_rows[1]
    ])
    monkeypatch.setattr(
        grid_editor5, "validate_bezier_control_nets",
        lambda *args, **kwargs: pytest.fail(
            "isolated curve ran patch Jacobian validation"
        ),
    )

    fixed = original[fixed_index]
    normal = np.array([-original_direction[1], original_direction[0]])
    new_length = 1.6 * patch.global_cap_tangent_length
    requested = fixed + new_length * original_direction + 8.0 * normal
    cap_handle.move_to_scene(patch.mapToScene(QPointF(*requested)))
    after_cap = np.array(patch.one_cap_global_curve_controls, copy=True)
    cap_vector = after_cap[cap_control_index] - fixed
    before_midpoint, unused_tangent = (
        grid_editor5.cubic_bezier_point_and_tangent_from_controls(
            original, 0.5
        )
    )
    after_midpoint, unused_tangent = (
        grid_editor5.cubic_bezier_point_and_tangent_from_controls(
            after_cap, 0.5
        )
    )
    assert np.isclose(patch.global_cap_tangent_length, new_length)
    assert np.isclose(np.dot(cap_vector, original_direction), new_length)
    assert abs(np.cross(cap_vector, original_direction)) < 2.e-12
    assert np.allclose(
        cap_vector / np.linalg.norm(cap_vector), original_direction
    )
    assert not np.allclose(after_midpoint, before_midpoint)
    assert np.allclose(qpath_cubic_controls(patch.path()), after_cap)
    outer_after_cap = np.array([
        grid_editor5.np_point(node.position) for node in patch.outer_nodes
    ])
    middle_after_cap = np.array([
        grid_editor5.np_point(node.position)
        for node in patch.preview_node_rows[1]
    ])
    assert not np.allclose(outer_after_cap[1:-1], outer_before_cap[1:-1])
    assert not np.allclose(middle_after_cap[1:-1], middle_before_cap[1:-1])

    endpoint_before = after_cap[endpoint_index]
    tangent_before = after_cap[tangent_index]
    outer_before_endpoint = np.array(outer_after_cap, copy=True)
    endpoint_delta = np.array([0.37, -0.42])
    endpoint_handle.move_to_scene(patch.mapToScene(
        grid_editor5.qt_point(endpoint_before + endpoint_delta)
    ))
    after_endpoint = np.array(patch.one_cap_global_curve_controls, copy=True)
    assert np.allclose(
        after_endpoint[endpoint_index], endpoint_before + endpoint_delta
    )
    assert np.allclose(
        after_endpoint[tangent_index], tangent_before + endpoint_delta
    )
    assert np.allclose(after_endpoint[fixed_index], fixed)
    assert np.allclose(qpath_cubic_controls(patch.path()), after_endpoint)
    outer_after_endpoint = np.array([
        grid_editor5.np_point(node.position) for node in patch.outer_nodes
    ])
    assert not np.allclose(outer_after_endpoint, outer_before_endpoint)

    tangent_target = after_endpoint[tangent_index] + np.array([-0.28, 0.51])
    outer_before_tangent = np.array(outer_after_endpoint, copy=True)
    tangent_handle.move_to_scene(patch.mapToScene(
        grid_editor5.qt_point(tangent_target)
    ))
    final_controls = np.array(patch.one_cap_global_curve_controls, copy=True)
    assert np.allclose(final_controls[tangent_index], tangent_target)
    assert np.allclose(
        final_controls[endpoint_index], after_endpoint[endpoint_index]
    )
    assert np.allclose(final_controls[fixed_index], fixed)
    assert np.array_equal(
        patch.global_cap_tangent_direction, original_direction
    )
    assert np.allclose(qpath_cubic_controls(patch.path()), final_controls)
    outer_after_tangent = np.array([
        grid_editor5.np_point(node.position) for node in patch.outer_nodes
    ])
    assert not np.allclose(
        outer_after_tangent[1:-1], outer_before_tangent[1:-1]
    )
    reverse_request = fixed - 100.0 * original_direction + 3.0 * normal
    cap_handle.move_to_scene(patch.mapToScene(QPointF(*reverse_request)))
    clamped = np.asarray(patch.one_cap_global_curve_controls)
    clamped_vector = clamped[cap_control_index] - fixed
    assert np.dot(clamped_vector, original_direction) > 0.0
    assert abs(np.cross(clamped_vector, original_direction)) < 2.e-12
    topology = patch.one_cap_topology
    cap_column = 0 if cap_at_start else -1
    cap_nodes = (
        topology.start_cap_nodes
        if cap_at_start else topology.end_cap_nodes
    )
    assert all(
        patch.preview_node_rows[index][cap_column] is cap_nodes[index]
        for index in range(1, patch.radial_layers + 1)
    )


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_isolated_one_cap_curve_initial_controls_use_global_chord_rule(
    monkeypatch, cap_at_start
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=3,
    )
    isolated = case.patch
    isolated.enable_bezier_mode(one_cap_curve_only=True)
    controls = np.asarray(isolated.one_cap_global_curve_controls)
    chord = controls[3] - controls[0]
    cap_vector = (
        controls[1] - controls[0]
        if cap_at_start else controls[2] - controls[3]
    )
    expected_free_control = (
        controls[3] - chord / 3.0
        if cap_at_start else controls[0] + chord / 3.0
    )

    assert np.isclose(np.linalg.norm(cap_vector), np.linalg.norm(chord) / 3.0)
    assert np.allclose(
        controls[2 if cap_at_start else 1], expected_free_control
    )
    assert np.array_equal(qpath_cubic_controls(isolated.path()), controls)


def synthetic_isolated_one_cap_curve(main_edge_count, cap_at_start):
    inner_nodes = [
        SimpleNamespace(
            index=index,
            position=QPointF(3.0 * index / main_edge_count, 0.0),
        )
        for index in range(main_edge_count + 1)
    ]
    inner_edges = [
        SimpleNamespace(uv_index=1) for unused in range(main_edge_count)
    ]
    fixed_x = 0.0 if cap_at_start else 3.0
    fixed_node = SimpleNamespace(
        index=1000 + main_edge_count,
        position=QPointF(fixed_x, 2.0),
        xx=np.array([[0.0, 1.0 / 3.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0 / 3.0, 0.0]]),
    )
    cap_edge = SimpleNamespace(element_index=9000 + main_edge_count)
    topology = SimpleNamespace(
        outer_start_node=fixed_node if cap_at_start else None,
        outer_end_node=None if cap_at_start else fixed_node,
        start_cap_edges=[cap_edge] if cap_at_start else [],
        end_cap_edges=[] if cap_at_start else [cap_edge],
        start_cap_nodes=[inner_nodes[0], fixed_node]
        if cap_at_start else [],
        end_cap_nodes=[] if cap_at_start
        else [inner_nodes[-1], fixed_node],
    )
    patch = grid_editor5.extended_patch(inner_nodes, inner_edges)
    patch.one_cap_topology = topology
    patch.outer_nodes = [
        grid_editor5.extended_patch_node(QPointF(
            3.0 * index / main_edge_count, 2.0
        ))
        for index in range(main_edge_count + 1)
    ]
    patch.enable_bezier_mode(one_cap_curve_only=True)
    return patch


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_isolated_global_curve_is_discretization_independent(cap_at_start):
    short = synthetic_isolated_one_cap_curve(4, cap_at_start)
    long = synthetic_isolated_one_cap_curve(40, cap_at_start)
    expected_roles = (
        {"end", "end_tangent", "cap_global_tangent"}
        if cap_at_start
        else {"start", "start_tangent", "cap_global_tangent"}
    )

    assert np.array_equal(
        short.one_cap_global_curve_controls,
        long.one_cap_global_curve_controls,
    )
    assert len(short.outer_nodes) == 5
    assert len(long.outer_nodes) == 41
    assert np.array_equal(
        qpath_cubic_controls(short.path()),
        qpath_cubic_controls(long.path()),
    )
    assert {handle.role for handle in short.bezier_handles} == expected_roles
    assert {handle.role for handle in long.bezier_handles} == expected_roles
    short_handles = {handle.role: handle for handle in short.bezier_handles}
    long_handles = {handle.role: handle for handle in long.bezier_handles}
    assert all(
        short_handles[role].pos() == long_handles[role].pos()
        for role in expected_roles
    )


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_one_cap_global_tangent_handle_changes_whole_exact_curve(
    monkeypatch, cap_at_start
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=3,
    )
    patch = case.patch
    patch.enable_bezier_mode()
    handles = {handle.role: handle for handle in patch.bezier_handles}
    free_handle = handles[
        "end_tangent" if cap_at_start else "start_tangent"
    ]
    free_handle.move_to_scene(
        free_handle.mapToScene(QPointF(0.0, 0.0)) + QPointF(0.0, 0.65)
    )
    initial_length = patch.global_cap_tangent_length
    original_rows = {
        index: {
            "controls": np.array(data["controls"], copy=True),
            "segments": np.array(data["segments"], copy=True),
            "points": np.array(data["points"], copy=True),
            "tangents": np.array(data["tangents"], copy=True),
        }
        for index, data in patch.one_cap_global_rows.items()
    }

    for multiplier in (0.6, 2.5):
        unused_handle, accepted = drag_cap_global_handle(
            patch, cap_at_start, multiplier * initial_length,
            lateral=5.0,
        )
        assert accepted
        for radial_index, data in patch.one_cap_global_rows.items():
            controls = np.asarray(data["controls"])
            parameters = np.asarray(data["parameters"])
            for index, segment in enumerate(data["segments"]):
                for local_parameter in (0.0, 0.17, 0.5, 0.83, 1.0):
                    parameter = (
                        parameters[index]
                        + local_parameter
                        * (parameters[index + 1] - parameters[index])
                    )
                    global_point, unused_tangent = (
                        grid_editor5.cubic_bezier_point_and_tangent_from_controls(
                            controls, parameter
                        )
                    )
                    local_point, unused_tangent = (
                        grid_editor5.cubic_bezier_point_and_tangent_from_controls(
                            segment, local_parameter
                        )
                    )
                    assert np.allclose(
                        local_point, global_point, atol=3.e-10
                    )

    for radial_index, original in original_rows.items():
        adjusted = patch.one_cap_global_rows[radial_index]
        cap_control = 1 if cap_at_start else 2
        other_control = 2 if cap_at_start else 1
        assert not np.allclose(
            adjusted["controls"][cap_control],
            original["controls"][cap_control],
        )
        assert np.allclose(
            adjusted["controls"][other_control],
            original["controls"][other_control],
        )
        assert not np.allclose(
            adjusted["points"][1:-1], original["points"][1:-1]
        )
        assert not np.allclose(
            adjusted["tangents"][1:-1], original["tangents"][1:-1]
        )
        # A later segment changes too; this cannot be a first-element edit.
        assert not np.allclose(
            adjusted["segments"][-1], original["segments"][-1]
        )


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_invalid_one_cap_global_tangent_drag_reverts_atomically(
    monkeypatch, cap_at_start
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=3,
    )
    patch = case.patch
    patch.enable_bezier_mode()
    initial_length = patch.global_cap_tangent_length
    unused_handle, accepted = drag_cap_global_handle(
        patch, cap_at_start, 3.0 * initial_length
    )
    assert accepted
    previous_length = patch.global_cap_tangent_length
    previous_rows = {
        index: np.array(data["controls"], copy=True)
        for index, data in patch.one_cap_global_rows.items()
    }
    previous_positions = [
        [grid_editor5.np_point(node.position) for node in row]
        for row in patch.preview_node_rows
    ]
    status = []
    case.view.set_patch_status = status.append

    handle, accepted = drag_cap_global_handle(
        patch, cap_at_start, 100.0 * initial_length, lateral=20.0
    )

    assert not accepted
    assert patch.one_cap_global_active
    assert patch.global_cap_tangent_length == previous_length
    assert patch.global_cap_tangent_last_valid_length == previous_length
    assert status == ["Invalid patch geometry"]
    assert all(
        np.array_equal(data["controls"], previous_rows[index])
        for index, data in patch.one_cap_global_rows.items()
    )
    assert all(
        np.array_equal(grid_editor5.np_point(node.position), expected)
        for row, expected_row in zip(
            patch.preview_node_rows, previous_positions
        )
        for node, expected in zip(row, expected_row)
    )
    fixed = (
        grid_editor5.np_point(patch.bezier_start_position())
        if cap_at_start
        else grid_editor5.np_point(patch.bezier_end_position())
    )
    assert np.isclose(
        np.linalg.norm(grid_editor5.np_point(handle.pos()) - fixed),
        previous_length,
    )


@pytest.mark.parametrize("cap_at_start", [True, False])
def test_one_cap_global_rows_drive_preview_and_committed_edges_exactly(
    monkeypatch, cap_at_start
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=2, cap_at_start=cap_at_start,
        main_edge_count=3,
    )
    cap_snapshots = [
        (grid_editor5.np_point(node.position), np.array(node.xx, copy=True))
        for node in case.cap_nodes
    ]
    patch = case.patch
    patch.enable_bezier_mode()
    handles = {handle.role: handle for handle in patch.bezier_handles}
    free_tangent_role = "end_tangent" if cap_at_start else "start_tangent"
    handle = handles[free_tangent_role]
    handle.move_to_scene(
        handle.mapToScene(QPointF(0.0, 0.0)) + QPointF(0.0, 0.65)
    )
    initial_cap_length = patch.global_cap_tangent_length
    unused_handle, accepted = drag_cap_global_handle(
        patch, cap_at_start, 1.5 * initial_cap_length, lateral=4.0
    )

    assert accepted
    assert np.isclose(
        patch.global_cap_tangent_length, 1.5 * initial_cap_length
    )
    assert patch.one_cap_global_active
    assert patch.one_cap_global_minimum_jacobian > 0.0
    for radial_index in range(1, patch.radial_layers + 1):
        tangents, parameters, segments = (
            grid_editor5.cap_first_global_bezier_row_data(
                patch, patch.one_cap_topology, radial_index
            )
        )
        row = (
            patch.preview_node_rows[radial_index]
            if cap_at_start
            else list(reversed(patch.preview_node_rows[radial_index]))
        )
        controls = np.asarray(segments[0])
        # Reconstruct the global controls from the retained local endpoint data.
        original = patch.one_cap_global_rows[radial_index]["controls"]
        if not cap_at_start:
            original = original[::-1]
        for node, parameter in zip(row, parameters):
            point, unused_tangent = (
                grid_editor5.cubic_bezier_point_and_tangent_from_controls(
                    original, parameter
                )
            )
            assert np.allclose(grid_editor5.np_point(node.position), point)
        for index, segment in enumerate(segments):
            assert np.allclose(segment[0], grid_editor5.np_point(row[index].position))
            assert np.allclose(
                segment[3], grid_editor5.np_point(row[index + 1].position)
            )
        first_local = segments[0][1] - segments[0][0]
        assert np.allclose(
            first_local,
            parameters[1] * tangents[0] / 3.0,
            atol=2.e-10,
        )
        left = segments[0][2] - segments[0][3]
        right = segments[1][1] - segments[1][0]
        assert abs(np.cross(left, right)) < 2.e-10
        assert np.dot(left, right) < 0.0
        assert np.isclose(
            np.linalg.norm(left) / np.linalg.norm(right),
            (parameters[1] - parameters[0])
            / (parameters[2] - parameters[1]),
        )

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(patch)
    for node, (position, xx) in zip(case.cap_nodes, cap_snapshots):
        assert np.array_equal(grid_editor5.np_point(node.position), position)
        assert np.array_equal(node.xx, xx)

    created_edges = {edge for element in elements for edge in element.edges}
    unequal_size_pair_seen = False
    for radial_index in range(1, len(node_rows)):
        unused_tangents, unused_parameters, segments = (
            grid_editor5.cap_first_global_bezier_row_data(
                patch, patch.one_cap_topology, radial_index
            )
        )
        for column, segment in enumerate(segments):
            edge = next(
                edge for edge in created_edges
                if frozenset(edge.vertices) == frozenset((
                    node_rows[radial_index][column].index,
                    node_rows[radial_index][column + 1].index,
                ))
            )
            controls = edge_bezier_points(edge.points)
            controls = np.column_stack((
                controls[:, 0, 0], controls[:, 1, 0],
                controls[:, 1, 1], controls[:, 0, 1],
            ))
            if edge.nodes[0] is not node_rows[radial_index][column]:
                controls = controls[:, ::-1]
            assert np.allclose(controls, np.asarray(segment).T, atol=2.e-10)
            unequal_size_pair_seen |= not np.isclose(
                abs(edge.sizes[1, 0]), abs(edge.sizes[1, 1])
            )
    assert unequal_size_pair_seen

    actual_nets = [element_bezier_points(element.points, 1.0) for element in elements]
    valid, minimum, unused_details = grid_editor5.validate_bezier_control_nets(
        actual_nets
    )
    assert valid
    assert minimum > 0.0
    assert np.isclose(minimum, patch.one_cap_global_minimum_jacobian)


def test_invalid_one_cap_global_candidate_falls_back_atomically(
    monkeypatch, capsys
):
    case = build_multi_layer_one_cap_case(
        monkeypatch, radial_layers=3, cap_at_start=True, main_edge_count=2
    )
    attach_old_cap_neighbors(
        case, case.cap_edges, case.cap_nodes,
        [0.5, 1.0, 2.0], [1.0, 0.0],
    )
    baseline_positions = [
        [grid_editor5.np_point(node.position) for node in row]
        for row in case.patch.preview_node_rows
    ]
    case.patch.enable_bezier_mode()

    assert not case.patch.one_cap_global_active
    assert case.patch.one_cap_global_rows == {}
    assert "using previous geometry" in capsys.readouterr().out
    assert all(
        np.allclose(grid_editor5.np_point(node.position), expected)
        for row, expected_row in zip(
            case.patch.preview_node_rows, baseline_positions
        )
        for node, expected in zip(row, expected_row)
    )

    node_rows, elements = grid_editor5.add_extended_patch_to_nodes_elements(
        case.patch
    )
    assert len(node_rows) == 4
    assert len(elements) == 6
