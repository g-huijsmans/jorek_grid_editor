import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide2.QtCore import QPoint, QPointF, Qt
from PySide2.QtGui import QBrush, QColor, QPainterPath, QPen, QTransform
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


def test_resize_uniformly_scales_current_view_and_preserves_center():
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
    app.processEvents()
    old_size = view.viewport().size()
    old_center = view.mapToScene(view.viewport().rect().center())
    old_scale = abs(view.transform().m11())

    view.resize(600, 400)
    app.processEvents()

    new_size = view.viewport().size()
    expected_factor = min(
        new_size.width() / old_size.width(),
        new_size.height() / old_size.height(),
    )
    transform = view.transform()
    new_center = view.mapToScene(view.viewport().rect().center())
    assert abs(transform.m11()) == pytest.approx(
        old_scale * expected_factor, rel=1e-6
    )
    assert abs(transform.m11()) == pytest.approx(abs(transform.m22()))
    assert transform.m22() < 0.0
    assert new_center.x() == pytest.approx(old_center.x(), abs=1.0)
    assert new_center.y() == pytest.approx(old_center.y(), abs=1.0)
    assert view.zoom_level == pytest.approx(abs(transform.m11()))
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
    assert "fit grid to window" in capsys.readouterr().out
    view.close()


def test_extended_patch_number_key_rebuilds_radial_preview_rows(
    monkeypatch, capsys
):
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
    event = type("KeyEvent", (), {"key": lambda self: Qt.Key_Escape})()

    view.keyPressEvent(event)

    assert view.selected_edges == []
    assert edge.pen().color() == Qt.yellow
    assert edge.pen().widthF() == 3.0
    assert view.current_patch is None
    assert patch.scene() is None
    assert view.selected_nodes == []
    assert view.selected_elements == []
    assert boundary_node_item.brush().color() == QColor(0, 0, 255)
    assert interior_node_item.brush().color() == QColor(255, 0, 0)
    assert element_path.brush().color() == QColor(255, 255, 255, 64)
    assert app is not None


def test_e_starts_same_direction_extended_patch_and_escape_cancels(
    monkeypatch, capsys
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
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
    assert first_edge.pen().color() == Qt.yellow
    assert second_edge.pen().color() == Qt.yellow
    assert app is not None


def test_e_previews_capped_boundary_gap():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
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


def test_one_cap_e_stays_manual_until_bezier_mode_is_requested(monkeypatch):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
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
    monkeypatch.setattr(
        view, "selected_main_uv_index_under_cursor", lambda: 1
    )

    view.keyPressEvent(event)

    patch = view.current_extended_patch
    assert patch.can_commit is True
    assert patch.bezier_mode is False
    assert patch.one_cap_topology.outer_start_node is nodes[0]
    assert patch.outer_nodes == []
    assert patch.add_outer_node(QPointF(2.0, 2.0))
    assert [node.position for node in patch.outer_nodes] == [
        QPointF(0.0, 2.0), QPointF(1.0, 2.0), QPointF(2.0, 2.0)
    ]
    assert not patch.add_outer_node(QPointF(3.0, 2.0))

    bezier_event = type("KeyEvent", (), {"key": lambda self: Qt.Key_B})()
    view.keyPressEvent(bezier_event)

    roles = {handle.role for handle in patch.bezier_handles}
    assert patch.outer_nodes[0] is nodes[0]
    assert roles == {"end", "end_tangent"}
    assert patch.bezier_start_position() == nodes[0].position
    assert np.allclose(patch.bezier_start_vector(), nodes[0].xx[:, 1])

    assert app is not None


def test_mixed_uv_topology_failure_does_not_fall_back_to_zero_cap(capsys):
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
    monkeypatch.setattr(
        view, "selected_main_uv_index_under_cursor", lambda: 1
    )
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
        "end", "end_tangent"
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


@pytest.mark.parametrize(
    "edge_count, radial_layers, bezier_mode",
    [
        (2, 1, False), (4, 1, False), (2, 2, False), (2, 4, False),
        (2, 1, True), (2, 4, True),
    ],
)
def test_extended_patch_creates_radial_rows(
    monkeypatch, edge_count, radial_layers, bezier_mode, capsys
):
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = this_view()
    view.setScene(scene)
    inner_count = edge_count + 1
    nodes_xx = np.zeros((2, 4, inner_count))
    for index in range(inner_count):
        nodes_xx[:,0,index] = [float(index), 0.0]
        nodes_xx[:,1,index] = [1.0, 0.0]
        nodes_xx[:,2,index] = [0.0, 1.0 / 3.0]
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
            edge_nodes, [index, index + 1], [0,1], 100 + index, 0, 1,
            grid_editor5.signed_edge_sizes(edge_nodes, 1),
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
    radial_uv_index = inner_edges[0].uv_index % 2 + 1
    if radial_layers > 1:
        assert diagnostic_output.count("main edge uv_index = 1") == (
            inner_count * radial_layers
        )
        assert diagnostic_output.count("perp_index = 2") == (
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
        assert np.all(np.diff(outer_edge[0,:]) >= 0)
        assert np.allclose(outer_edge[1,:], layer_height)
    assert all(edge.uv_index == 1 for edge in outer_edges)
    assert all(edge.uv_index == 2 for edge in all_transverse_edges)
    if bezier_mode:
        assert diagnostic_output.count("Bezier outer edge") == edge_count
        for node, tangent in zip(node_rows[-1], patch.outer_tangents):
            assert np.allclose(node.xx[:, 1], np.asarray(tangent) / 3.0)
            assert np.dot(node.xx[:, 1], np.asarray(tangent)) > 0.0
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
                np.array([dt, -dt])
                if edge.vertices[0] == final_indices[index]
                else np.array([-dt, dt])
            )
            assert np.allclose(edge.sizes[1, :], expected_sizes)
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

    def record_two_edges(*args):
        element = real_two_edges(*args)
        helper_calls.append(("two", element))
        return element

    def record_three_edges(*args):
        element = real_three_edges(*args)
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
