import sys
import os
import random
import string
import math
import tempfile
import time
import ctypes
import numpy as np
from PySide2.QtCore import (
    Qt, QEvent, QPoint, QPointF, QRect, QRectF, QSize, QTimer,
)
from PySide2.QtGui import (
    QPen, QFont, QBrush, QColor, QMouseEvent, QPainter, QPainterPath,
    QPainterPathStroker, QKeySequence, QTransform,
)
from PySide2.QtWidgets import (
    QAbstractSpinBox, QAction, QApplication, QComboBox, QFileDialog,
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
    QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QRubberBand, QSizePolicy,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from jorek             import *
from jorek_bezier      import edge_bezier_points, element_bezier_points
from jorek_geometry    import (
    basis_vector_from_scene,
    pair_corners_to_edge_endpoints,
    reorder_corners,
)

jorek_grid = jorek


NODE_MARKER_SIZE = 8.0
NODE_COLOR = QColor(60, 60, 60)
SELECTED_NODE_COLOR = QColor(0, 255, 0)
VECTOR_HANDLE_SIZE = 8.0
BASIS_VECTOR_WIDTH = 2.0
EXTENDED_BEZIER_HANDLE_SIZE = 9.0
EXTENDED_PATCH_LINE_WIDTH = 1.75
GRAPHICS_HANDLE_OUTLINE_WIDTH = 1.0
BOUNDARY_EDGE_WIDTH = 2
BOUNDARY_EDGE_COLOR = QColor(0, 255, 255)
STATIC_MESH_WIDTH = 0.75
Z_STATIC_MESH = -1.0
Z_WALL_OUTLINE = -0.25
Z_MESH_EDGE = 0.0
Z_BOUNDARY_EDGE = 0.5
Z_PATCH_PREVIEW = 1.0
Z_NODE = 2.0
Z_VECTOR_HANDLE = 4.0
WALL_OUTLINE_COLOR = QColor(128, 0, 128)
WALL_OUTLINE_WIDTH = 2.0
MAX_RADIAL_LAYERS = 99

this_scaling = 100.0
node_list = []
element_list = []
boundary_list = []
scene = None
view = None
static_mesh = None

MEMORY_DIAGNOSTICS = os.environ.get(
    "JOREK_GRID_MEMORY_DIAGNOSTICS", "0"
) == "1"
DIAGNOSTIC_ELEMENTS_ONLY = os.environ.get(
    "JOREK_GRID_DIAGNOSTIC_ELEMENTS_ONLY", "0"
) == "1"
DIAGNOSTIC_BASIS_SCALE = os.environ.get(
    "JOREK_GRID_DIAGNOSTIC_BASIS_SCALE", "0"
) == "1"
SHOW_EDGE_INDICES = os.environ.get(
    "JOREK_GRID_SHOW_EDGE_INDICES", "0"
) == "1"
_memory_diagnostic_records = []
_memory_diagnostic_previous_time = None
_memory_diagnostic_previous_rss = None
_memory_diagnostic_peak_rss = None
_diagnostic_nodes_xx_before = None


def boundary_edge_pen(color=None, width=BOUNDARY_EDGE_WIDTH):
    if color is None:
        color = BOUNDARY_EDGE_COLOR
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCosmetic(True)
    return pen


def graphics_handle_outline_pen():
    pen = QPen(Qt.black)
    pen.setWidthF(GRAPHICS_HANDLE_OUTLINE_WIDTH)
    pen.setCosmetic(True)
    return pen


def parse_wall_outline(filename):
    """Read ordered physical R/Z; columns after the first two are ignored."""
    points = []
    header_skipped = False
    with open(filename, "r") as wall_file:
        for line_number, raw_line in enumerate(wall_file, 1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "!")):
                continue
            columns = line.split()
            try:
                if len(columns) < 2:
                    raise ValueError
                r_value = float(columns[0])
                z_value = float(columns[1])
            except ValueError:
                if not points and not header_skipped and len(columns) >= 2:
                    try:
                        float(columns[0])
                        first_is_numeric = True
                    except ValueError:
                        first_is_numeric = False
                    try:
                        float(columns[1])
                        second_is_numeric = True
                    except ValueError:
                        second_is_numeric = False
                    if not first_is_numeric and not second_is_numeric:
                        header_skipped = True
                        continue
                raise ValueError(
                    "Invalid wall coordinates on line {}: expected at least "
                    "two numeric columns".format(line_number)
                )
            if not math.isfinite(r_value) or not math.isfinite(z_value):
                raise ValueError(
                    "Invalid wall coordinates on line {}: R and Z must be "
                    "finite".format(line_number)
                )
            points.append((r_value, z_value))
    if len(points) < 2:
        raise ValueError("Wall outline must contain at least two valid points")
    return points


class wall_outline_item(QGraphicsPathItem):
    """Lightweight, noninteractive ordered wall-reference polyline."""
    def __init__(self, physical_points, scaling=None):
        if scaling is None:
            scaling = this_scaling
        path = QPainterPath()
        first_r, first_z = physical_points[0]
        path.moveTo(scaling * first_r, scaling * first_z)
        for r_value, z_value in physical_points[1:]:
            path.lineTo(scaling * r_value, scaling * z_value)
        super().__init__(path)
        pen = QPen(WALL_OUTLINE_COLOR)
        pen.setWidthF(WALL_OUTLINE_WIDTH)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(Qt.NoBrush)
        self.setZValue(Z_WALL_OUTLINE)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.NoButton)


def graphics_item_in_scene(item, expected_scene):
    if item is None:
        return False
    try:
        return item.scene() is expected_scene
    except RuntimeError:
        return False


def _process_rss_bytes():
    global _memory_diagnostic_peak_rss
    try:
        import psutil
        memory_info = psutil.Process(os.getpid()).memory_info()
        _memory_diagnostic_peak_rss = getattr(
            memory_info, "peak_wset", _memory_diagnostic_peak_rss
        )
        return memory_info.rss
    except ImportError:
        if os.name != "nt":
            try:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return rss if sys.platform == "darwin" else rss * 1024
            except ImportError:
                return None

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            return None
        _memory_diagnostic_peak_rss = counters.PeakWorkingSetSize
        return counters.WorkingSetSize


def reset_memory_diagnostics():
    global _memory_diagnostic_previous_time, _memory_diagnostic_previous_rss
    global _memory_diagnostic_peak_rss
    _memory_diagnostic_records[:] = []
    _memory_diagnostic_peak_rss = None
    _memory_diagnostic_previous_time = time.perf_counter()
    _memory_diagnostic_previous_rss = _process_rss_bytes()


def report_memory(stage):
    """Temporary process/graphics checkpoint enabled only for diagnostics."""
    global _memory_diagnostic_previous_time, _memory_diagnostic_previous_rss
    if not MEMORY_DIAGNOSTICS:
        return
    now = time.perf_counter()
    rss_bytes = _process_rss_bytes()
    rss_mb = rss_bytes / 1024.0 ** 2 if rss_bytes is not None else float("nan")
    previous_rss = _memory_diagnostic_previous_rss
    delta_mb = (
        (rss_bytes - previous_rss) / 1024.0 ** 2
        if rss_bytes is not None and previous_rss is not None else float("nan")
    )
    elapsed = (
        now - _memory_diagnostic_previous_time
        if _memory_diagnostic_previous_time is not None else 0.0
    )
    current_scene = globals().get("scene")
    scene_items = current_scene.items() if current_scene is not None else []
    type_counts = {}
    for item in scene_items:
        name = type(item).__name__
        type_counts[name] = type_counts.get(name, 0) + 1
    nodes = globals().get("node_list", []) or []
    elements = globals().get("element_list", []) or []
    boundaries = globals().get("boundary_list", []) or []
    print(
        "[MEM] {} | RSS={:.1f} MB | delta={:+.1f} MB | time={:.3f} s".format(
            stage, rss_mb, delta_mb, elapsed
        ),
        flush=True,
    )
    print(
        "      nodes={} elements={} boundary_edges={} scene_items={}".format(
            len(nodes), len(elements), len(boundaries), len(scene_items)
        ),
        flush=True,
    )
    if type_counts:
        print(
            "      scene types: " + ", ".join(
                "{}={}".format(name, count)
                for name, count in sorted(type_counts.items())
            ),
            flush=True,
        )
    _memory_diagnostic_records.append((stage, rss_mb, delta_mb, elapsed))
    _memory_diagnostic_previous_time = now
    _memory_diagnostic_previous_rss = rss_bytes


def report_grid_array_memory(grid):
    if not MEMORY_DIAGNOSTICS:
        return
    arrays = {
        "x": grid.nodes_xx,
        "boundary": grid.boundary,
        "vertex": grid.vertices,
        "size": grid.elements_size,
    }
    total = sum(np.asarray(array).nbytes for array in arrays.values())
    print(
        "[MEM] raw grid arrays: {} = {:.2f} MB".format(
            ", ".join(
                "{} {:.2f} MB".format(
                    name, np.asarray(array).nbytes / 1024.0 ** 2
                )
                for name, array in arrays.items()
            ),
            total / 1024.0 ** 2,
        ),
        flush=True,
    )


def report_boundary_diagnostics():
    if not MEMORY_DIAGNOSTICS:
        return
    boundary_nodes = {
        node.index for node in node_list if node.boundary != 0
    }
    touching_elements = {
        element.index for element in element_list
        if any(vertex in boundary_nodes for vertex in element.vertices)
    }
    elements_with_edges = {
        edge.element_index for edge in boundary_list
        if edge.element_index is not None
    }
    print(
        "[MEM] boundary diagnostics: boundary_nodes={} "
        "boundary_edges={} touching_elements={} elements_with_edges={}".format(
            len(boundary_nodes), len(boundary_list),
            len(touching_elements), len(elements_with_edges),
        ),
        flush=True,
    )


def report_graphics_multiplication():
    if not MEMORY_DIAGNOSTICS:
        return
    node_count = sum(
        isinstance(node, jorek_node_item) for node in node_list
    )
    element_count = sum(
        isinstance(element, jorek_element_item) for element in element_list
    )
    print(
        "[MEM] graphics multiplication: per node = 1 jorek_node_item + "
        "1 ellipse_item + 2 basis_vector_handle; implied node objects = {}".format(
            4 * node_count
        ),
        flush=True,
    )
    print(
        "[MEM] graphics multiplication: per element = 1 jorek_element_item + "
        "1 path_item; implied element objects = {}".format(2 * element_count),
        flush=True,
    )


def print_memory_diagnostic_table():
    if not MEMORY_DIAGNOSTICS:
        return
    print("\n[MEM] summary", flush=True)
    print("stage | RSS MB | delta MB | time s", flush=True)
    for stage, rss_mb, delta_mb, elapsed in _memory_diagnostic_records:
        print(
            "{} | {:.1f} | {:+.1f} | {:.3f}".format(
                stage, rss_mb, delta_mb, elapsed
            ),
            flush=True,
        )
    if _memory_diagnostic_records:
        biggest = max(_memory_diagnostic_records, key=lambda record: record[2])
        print(
            "[MEM] biggest checkpoint jump: {} ({:+.1f} MB)".format(
                biggest[0], biggest[2]
            ),
            flush=True,
        )
    if _memory_diagnostic_peak_rss is not None:
        print(
            "[MEM] process peak RSS: {:.1f} MB".format(
                _memory_diagnostic_peak_rss / 1024.0 ** 2
            ),
            flush=True,
        )


def _ratio_statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), float("nan")
    return tuple(np.percentile(values, [50.0, 90.0, 100.0]))


def report_basis_scale_diagnostics(grid):
    """Compare raw nodal derivatives with element-local control vectors."""
    if not DIAGNOSTIC_BASIS_SCALE:
        return

    positions = grid.nodes_xx[:, 0, grid.vertices]
    raw_u_vectors = grid.nodes_xx[:, 1, grid.vertices]
    raw_v_vectors = grid.nodes_xx[:, 2, grid.vertices]
    u_neighbor = np.array([1, 0, 3, 2])
    v_neighbor = np.array([3, 2, 1, 0])
    chord_u = np.linalg.norm(
        positions - positions[:, u_neighbor, :], axis=0
    )
    chord_v = np.linalg.norm(
        positions - positions[:, v_neighbor, :], axis=0
    )
    raw_u = np.linalg.norm(raw_u_vectors, axis=0)
    raw_v = np.linalg.norm(raw_v_vectors, axis=0)
    effective_u = np.linalg.norm(
        raw_u_vectors * grid.elements_size[1, :, :][None, :, :], axis=0
    )
    effective_v = np.linalg.norm(
        raw_v_vectors * grid.elements_size[2, :, :][None, :, :], axis=0
    )
    valid_u = chord_u > np.finfo(float).eps
    valid_v = chord_v > np.finfo(float).eps
    ratios = {
        "raw_u / chord_u": raw_u[valid_u] / chord_u[valid_u],
        "effective_u / chord_u": effective_u[valid_u] / chord_u[valid_u],
        "raw_v / chord_v": raw_v[valid_v] / chord_v[valid_v],
        "effective_v / chord_v": effective_v[valid_v] / chord_v[valid_v],
    }
    print("[BASIS] aggregate ratios (median, p90, maximum)", flush=True)
    for name, values in ratios.items():
        median, p90, maximum = _ratio_statistics(values)
        print(
            "        {:26s} {:10.4g} {:10.4g} {:10.4g}".format(
                name, median, p90, maximum
            ),
            flush=True,
        )

    centroids = positions.mean(axis=1)
    center = np.median(centroids, axis=1)
    boundary_elements = np.any(grid.boundary[grid.vertices] != 0, axis=0)
    interior_indices = np.flatnonzero(~boundary_elements)
    boundary_indices = np.flatnonzero(boundary_elements)

    samples = [(
        "central/core proxy",
        int(np.argmin(np.linalg.norm(centroids - center[:, None], axis=0))),
    )]
    if interior_indices.size:
        samples.append((
            "outer-R interior proxy",
            int(interior_indices[np.argmax(centroids[0, interior_indices])]),
        ))
    samples.extend([
        ("lower/divertor proxy", int(np.argmin(centroids[1]))),
        ("upper/SOL proxy", int(np.argmax(centroids[1]))),
    ])
    if boundary_indices.size:
        samples.extend([
            ("boundary outer-R", int(
                boundary_indices[np.argmax(centroids[0, boundary_indices])]
            )),
            ("boundary lower", int(
                boundary_indices[np.argmin(centroids[1, boundary_indices])]
            )),
        ])

    unique_samples = []
    seen = set()
    for label, element_index in samples:
        if element_index not in seen:
            unique_samples.append((label, element_index))
            seen.add(element_index)

    print("[BASIS] representative element vertices", flush=True)
    for label, element_index in unique_samples:
        print(
            "[BASIS] {}: element {} centroid=({:.6g}, {:.6g})".format(
                label, element_index,
                centroids[0, element_index], centroids[1, element_index],
            ),
            flush=True,
        )
        for local_vertex, node_index in enumerate(
            grid.vertices[:, element_index]
        ):
            cu = chord_u[local_vertex, element_index]
            cv = chord_v[local_vertex, element_index]
            ru = raw_u[local_vertex, element_index]
            rv = raw_v[local_vertex, element_index]
            eu = effective_u[local_vertex, element_index]
            ev = effective_v[local_vertex, element_index]
            print(
                "        vertex {} node {} | "
                "u chord={:.5g} raw={:.5g} effective={:.5g} "
                "ratios=({:.4g}, {:.4g}) | "
                "v chord={:.5g} raw={:.5g} effective={:.5g} "
                "ratios=({:.4g}, {:.4g})".format(
                    local_vertex, node_index,
                    cu, ru, eu, ru / cu, eu / cu,
                    cv, rv, ev, rv / cv, ev / cv,
                ),
                flush=True,
            )

    print("[BASIS] element_bezier_points control-vector checks", flush=True)
    for unused_label, element_index in unique_samples[:4]:
        vertices = grid.vertices[:, element_index]
        raw_points = grid.nodes_xx[:, :, vertices]
        scaled_points = raw_points.copy()
        scaled_points *= grid.elements_size[:, :, element_index][None, :, :]
        control_points = element_bezier_points(scaled_points, this_scaling)
        actual_u = control_points[:, 1, 0] - control_points[:, 0, 0]
        actual_v = control_points[:, 0, 1] - control_points[:, 0, 0]
        expected_u = this_scaling * scaled_points[:, 1, 0]
        expected_v = this_scaling * scaled_points[:, 2, 0]
        print(
            "        element {} vertex 0: raw_u={:.5g} scaled_u={:.5g} "
            "control_error_u={:.3g}; raw_v={:.5g} scaled_v={:.5g} "
            "control_error_v={:.3g}".format(
                element_index,
                np.linalg.norm(raw_points[:, 1, 0]),
                np.linalg.norm(scaled_points[:, 1, 0]),
                np.linalg.norm(actual_u - expected_u),
                np.linalg.norm(raw_points[:, 2, 0]),
                np.linalg.norm(scaled_points[:, 2, 0]),
                np.linalg.norm(actual_v - expected_v),
            ),
            flush=True,
        )

    element_selection = grid.nodes_xx[:, :, grid.vertices[:, 0]]
    boundary_vertices = grid.vertices[:2, 0].tolist()
    boundary_selection = (
        grid.nodes_xx[:, 0:2, boundary_vertices] * this_scaling
    )
    print(
        "[BASIS] NumPy aliasing: element advanced indexing shares memory={} ; "
        "boundary advanced indexing/scaling shares memory={}".format(
            np.shares_memory(element_selection, grid.nodes_xx),
            np.shares_memory(boundary_selection, grid.nodes_xx),
        ),
        flush=True,
    )


def verify_diagnostic_grid_unchanged(grid, stage):
    if not DIAGNOSTIC_BASIS_SCALE or _diagnostic_nodes_xx_before is None:
        return
    unchanged = np.array_equal(grid.nodes_xx, _diagnostic_nodes_xx_before)
    print(
        "[BASIS] nodes_xx unchanged {}: {}".format(stage, unchanged),
        flush=True,
    )
    assert unchanged, "Diagnostic detected in-place modification of nodes_xx"


def validate_single_element_patch(selected_edges):
    if any(not getattr(edge, "active", True) for edge in selected_edges):
        return "Inactive boundary edges cannot define a patch"
    if len(selected_edges) == 0:
        return "Select one or two boundary edges before starting a patch"
    if len(selected_edges) == 1:
        return None
    if len(selected_edges) > 3:
        return "This operation supports at most three selected boundary edges"

    if len(selected_edges) == 3:
        try:
            ordered_three_edge_chain(selected_edges)
        except ValueError as error:
            return str(error)
        return None

    edge0, edge1 = selected_edges
    shared_nodes = {
        node.index for node in edge0.nodes
    }.intersection(node.index for node in edge1.nodes)
    if len(shared_nodes) != 1:
        return (
            "The selected boundary edges must share exactly one node"
        )
    if edge0.uv_index == edge1.uv_index:
        return (
            "The selected boundary edges must use different local directions "
            "(one u-edge and one v-edge)"
        )
    if edge0.element_index == edge1.element_index:
        return "The two selected boundary edges must belong to different elements"
    return None


def patch_corner_count_error(edges, corner_nodes):
    edge_count = len(edges)
    corner_count = len(corner_nodes)
    if edge_count == 1 and corner_count != 2:
        return "One selected boundary edge requires exactly two new points"
    if edge_count == 2:
        selection_error = validate_single_element_patch(edges)
        if selection_error:
            return selection_error
        if corner_count != 1:
            return "Two selected boundary edges require exactly one new point"
        return None
    if edge_count == 3:
        selection_error = validate_single_element_patch(edges)
        if selection_error:
            return selection_error
        if corner_count != 0:
            return "Three selected boundary edges require no new points"
        return None
    if edge_count not in (1, 2, 3):
        return "A patch requires one, two, or three selected boundary edges"
    return None


def ordered_edge_chain(edges):
    if not edges:
        raise ValueError("At least one boundary edge is required")
    if any(not getattr(edge, "active", True) for edge in edges):
        raise ValueError("Inactive boundary edges cannot define a chain")

    nodes_by_index = {}
    edges_by_node = {}
    edge_keys = set()
    for edge in edges:
        if len(edge.nodes) != 2 or edge.nodes[0].index == edge.nodes[1].index:
            raise ValueError("Each selected boundary edge must connect two distinct vertices")
        edge_key = frozenset((edge.nodes[0].index, edge.nodes[1].index))
        if edge_key in edge_keys:
            raise ValueError("Duplicate boundary edges are not allowed")
        edge_keys.add(edge_key)
        for node in edge.nodes:
            nodes_by_index.setdefault(node.index, node)
            edges_by_node.setdefault(node.index, []).append(edge)

    endpoint_indices = sorted(
        index for index, incident_edges in edges_by_node.items()
        if len(incident_edges) == 1
    )
    if len(endpoint_indices) != 2:
        raise ValueError(
            "Selected boundary edges must form an open chain with exactly two endpoints"
        )
    if any(
        len(incident_edges) != 2
        for index, incident_edges in edges_by_node.items()
        if index not in endpoint_indices
    ):
        raise ValueError(
            "Every intermediate boundary-chain vertex must have degree 2"
        )

    ordered_nodes = [nodes_by_index[endpoint_indices[0]]]
    ordered_edges = []
    used_edge_ids = set()
    current_index = endpoint_indices[0]
    while current_index != endpoint_indices[1]:
        candidates = [
            edge for edge in edges_by_node[current_index]
            if id(edge) not in used_edge_ids
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Selected boundary edges must form one connected non-branching chain"
            )
        edge = candidates[0]
        used_edge_ids.add(id(edge))
        ordered_edges.append(edge)
        next_node = next(node for node in edge.nodes if node.index != current_index)
        ordered_nodes.append(next_node)
        current_index = next_node.index

    if len(ordered_edges) != len(edges):
        raise ValueError(
            "Selected boundary edges must form one connected open chain"
        )
    return ordered_nodes, ordered_edges


def validate_boundary_chain(edges):
    if len(edges) < 2:
        return "A boundary-chain operation requires at least two selected edges"
    try:
        ordered_edge_chain(edges)
    except ValueError as error:
        return str(error)
    return None


class extended_boundary_topology:
    def __init__(
        self, inner_nodes, inner_edges, outer_start, outer_end,
        start_side_edge=None, end_side_edge=None,
        start_cap_edges=None, end_cap_edges=None,
        start_cap_nodes=None, end_cap_nodes=None,
    ):
        self.inner_nodes = inner_nodes
        self.inner_edges = inner_edges
        self.outer_start = outer_start
        self.outer_end = outer_end

        self.start_cap_edges = list(
            start_cap_edges or ([start_side_edge] if start_side_edge else [])
        )
        self.end_cap_edges = list(
            end_cap_edges or ([end_side_edge] if end_side_edge else [])
        )
        self.start_cap_nodes = list(start_cap_nodes or (
            [inner_nodes[0], outer_start] if outer_start is not None else []
        ))
        self.end_cap_nodes = list(end_cap_nodes or (
            [inner_nodes[-1], outer_end] if outer_end is not None else []
        ))

        # Compatibility with the existing one-edge cap insertion code.
        self.start_cap_edge = (
            self.start_cap_edges[0] if self.start_cap_edges else None
        )
        self.end_cap_edge = (
            self.end_cap_edges[0] if self.end_cap_edges else None
        )
        self.start_side_edge = self.start_cap_edge
        self.end_side_edge = self.end_cap_edge
        self.outer_start_node = (
            self.start_cap_nodes[-1] if self.start_cap_nodes else None
        )
        self.outer_end_node = (
            self.end_cap_nodes[-1] if self.end_cap_nodes else None
        )


def ordered_extended_boundary_topology(edges, main_uv_index=None):
    if not edges:
        raise ValueError("Select at least one boundary edge")
    uv_indices = {edge.uv_index for edge in edges}
    if len(uv_indices) > 2:
        raise ValueError(
            "Extended boundary topology supports at most two local directions"
        )
    if main_uv_index is not None and main_uv_index not in uv_indices:
        raise ValueError(
            "The requested main boundary direction is not selected"
        )

    def diagnostic_vertices(edge):
        return list(getattr(
            edge, "vertices", [node.index for node in edge.nodes]
        ))

    candidates = []
    rejection_reasons = []
    candidate_uv_indices = (
        [main_uv_index] if main_uv_index is not None else uv_indices
    )
    for candidate_main_uv_index in candidate_uv_indices:
        main_edges = [
            edge for edge in edges
            if edge.uv_index == candidate_main_uv_index
        ]
        side_edges = [
            edge for edge in edges
            if edge.uv_index != candidate_main_uv_index
        ]
        print("candidate main uv:", candidate_main_uv_index)
        print("main edges:", [diagnostic_vertices(edge) for edge in main_edges])
        print("side edges:", [diagnostic_vertices(edge) for edge in side_edges])
        if not main_edges:
            reason = "candidate has no main edges"
            print("candidate rejected:", reason)
            rejection_reasons.append(reason)
            continue
        try:
            inner_nodes, inner_edges = ordered_edge_chain(main_edges)
        except ValueError as error:
            reason = "main edges do not form an open chain: {}".format(error)
            print("candidate rejected:", reason)
            rejection_reasons.append(reason)
            continue
        endpoint_indices = [inner_nodes[0].index, inner_nodes[-1].index]
        matched_side_chains = [[], []]
        matched_node_chains = [[], []]
        inner_node_indices = {node.index for node in inner_nodes}
        remaining_side_edges = list(side_edges)
        valid_candidate = True
        while remaining_side_edges:
            component = [remaining_side_edges.pop()]
            component_nodes = {node.index for node in component[0].nodes}
            changed = True
            while changed:
                changed = False
                for edge in list(remaining_side_edges):
                    if any(node.index in component_nodes for node in edge.nodes):
                        remaining_side_edges.remove(edge)
                        component.append(edge)
                        component_nodes.update(node.index for node in edge.nodes)
                        changed = True
            attached_slots = [
                slot for slot, endpoint in enumerate(endpoint_indices)
                if endpoint in component_nodes
            ]
            if len(attached_slots) != 1:
                reason = "cap chain does not attach to exactly one main-chain endpoint"
                print("candidate rejected:", reason)
                rejection_reasons.append(reason)
                valid_candidate = False
                break
            slot = attached_slots[0]
            if matched_side_chains[slot]:
                reason = "two cap chains attach to the same endpoint"
                print("candidate rejected:", reason)
                rejection_reasons.append(reason)
                valid_candidate = False
                break
            if any(
                index in inner_node_indices and index != endpoint_indices[slot]
                for index in component_nodes
            ):
                reason = "cap chain contains a non-endpoint main-chain node"
                print("candidate rejected:", reason)
                rejection_reasons.append(reason)
                valid_candidate = False
                break
            try:
                cap_nodes, cap_edges = ordered_edge_chain(component)
            except ValueError as error:
                reason = "cap edges do not form an open non-branching chain: {}".format(error)
                print("candidate rejected:", reason)
                rejection_reasons.append(reason)
                valid_candidate = False
                break
            if cap_nodes[-1].index == endpoint_indices[slot]:
                cap_nodes.reverse()
                cap_edges.reverse()
            if cap_nodes[0].index != endpoint_indices[slot]:
                reason = "cap chain does not begin at its main-chain endpoint"
                print("candidate rejected:", reason)
                rejection_reasons.append(reason)
                valid_candidate = False
                break
            matched_side_chains[slot] = cap_edges
            matched_node_chains[slot] = cap_nodes
        if not valid_candidate:
            continue
        defined_outer_indices = [chain[-1].index for chain in matched_node_chains if chain]
        if len(defined_outer_indices) != len(set(defined_outer_indices)):
            reason = "side edges connect to the same outer endpoint node"
            print("candidate rejected:", reason)
            rejection_reasons.append(reason)
            continue
        candidates.append(extended_boundary_topology(
            inner_nodes, inner_edges,
            matched_node_chains[0][-1] if matched_node_chains[0] else None,
            matched_node_chains[1][-1] if matched_node_chains[1] else None,
            start_cap_edges=matched_side_chains[0],
            end_cap_edges=matched_side_chains[1],
            start_cap_nodes=matched_node_chains[0],
            end_cap_nodes=matched_node_chains[1],
        ))

    if not candidates:
        raise ValueError(
            "Selected edges do not form a valid extended boundary topology: "
            + "; ".join(rejection_reasons)
        )
    if len(candidates) > 1:
        print("candidate rejected: ambiguous candidate")
        raise ValueError(
            "Ambiguous extended patch: specify which boundary direction "
            "is the main boundary"
        )
    return candidates[0]


def ordered_capped_boundary_gap(edges):
    topology = ordered_extended_boundary_topology(edges)
    if topology.start_cap_edge is None or topology.end_cap_edge is None:
        raise ValueError("A capped boundary gap requires two end edges")
    return topology


def ordered_three_edge_chain(edges):
    if len(edges) != 3:
        raise ValueError("Exactly three boundary edges are required")
    if len({node.index for edge in edges for node in edge.nodes}) != 4:
        raise ValueError("Three selected boundary edges must contain exactly four distinct vertices")

    ordered_nodes, ordered_edges = ordered_edge_chain(edges)
    if any(
        ordered_edges[i].uv_index == ordered_edges[i + 1].uv_index
        for i in range(2)
    ):
        raise ValueError("Adjacent edges in the three-edge chain must alternate uv_index")
    return ordered_nodes, ordered_edges


def two_edge_corner_nodes(edge0, edge1):
    shared_node_indices = {
        node.index for node in edge0.nodes
    }.intersection(node.index for node in edge1.nodes)
    if len(shared_node_indices) != 1:
        raise ValueError("Two boundary edges must share exactly one node")

    shared_node_index = next(iter(shared_node_indices))
    shared_node = next(
        node for node in edge0.nodes if node.index == shared_node_index
    )
    outer_node0 = next(
        node for node in edge0.nodes if node.index != shared_node_index
    )
    outer_node1 = next(
        node for node in edge1.nodes if node.index != shared_node_index
    )
    return shared_node, outer_node0, outer_node1


class extended_patch_node:
    def __init__(self, position, along_vector=None):
        self.position = position
        self.along_vector = (
            None if along_vector is None
            else np.asarray(along_vector, dtype=float)
        )


def order_outer_nodes(inner_nodes, outer_nodes):
    if len(inner_nodes) != len(outer_nodes):
        raise ValueError("Inner and outer node counts must match")
    if len(outer_nodes) < 2:
        return list(outer_nodes)

    inner_start = np_point(inner_nodes[0].position)
    inner_end = np_point(inner_nodes[-1].position)

    def position_key(node):
        return (node.position.x(), node.position.y())

    endpoint_candidates = []
    for start_index, start_node in enumerate(outer_nodes):
        for end_index, end_node in enumerate(outer_nodes):
            if start_index == end_index:
                continue
            cost = (
                np.linalg.norm(np_point(start_node.position) - inner_start)
                + np.linalg.norm(np_point(end_node.position) - inner_end)
            )
            endpoint_candidates.append((
                cost, position_key(start_node), position_key(end_node),
                start_index, end_index,
            ))
    unused_cost, unused_start_key, unused_end_key, start_index, end_index = min(
        endpoint_candidates
    )
    start_node = outer_nodes[start_index]
    end_node = outer_nodes[end_index]
    outer_direction = np_point(end_node.position - start_node.position)
    direction_norm_squared = np.inner(outer_direction, outer_direction)

    remaining_nodes = [
        node for index, node in enumerate(outer_nodes)
        if index not in (start_index, end_index)
    ]

    def projection_key(node):
        relative_position = np_point(node.position - start_node.position)
        if direction_norm_squared == 0:
            projection = 0.0
        else:
            projection = np.inner(
                relative_position, outer_direction
            ) / direction_norm_squared
        return (projection,) + position_key(node)

    remaining_nodes.sort(key=projection_key)
    return [start_node] + remaining_nodes + [end_node]


def interpolate_positions(start, end, segment_count):
    return [
        QPointF(
            (1. - index / segment_count) * start.x()
            + index / segment_count * end.x(),
            (1. - index / segment_count) * start.y()
            + index / segment_count * end.y(),
        )
        for index in range(segment_count + 1)
    ]


def cumulative_node_length_fractions(nodes):
    positions = [np_point(node.position) for node in nodes]
    lengths = [
        np.linalg.norm(positions[index + 1] - positions[index])
        for index in range(len(positions) - 1)
    ]
    total = sum(lengths)
    if total == 0:
        return np.linspace(0., 1., len(nodes))
    return np.array([0.] + list(np.cumsum(lengths) / total))


def cubic_bezier_point_and_tangent(start, end, start_vector, end_vector, t):
    control0 = start
    control1 = start + start_vector
    control2 = end + end_vector
    control3 = end
    one_minus_t = 1. - t
    point = (
        one_minus_t ** 3 * control0
        + 3. * one_minus_t ** 2 * t * control1
        + 3. * one_minus_t * t ** 2 * control2
        + t ** 3 * control3
    )
    tangent = (
        3. * one_minus_t ** 2 * (control1 - control0)
        + 6. * one_minus_t * t * (control2 - control1)
        + 3. * t ** 2 * (control3 - control2)
    )
    return point, tangent


def cubic_bezier_controls(start, end, start_vector, end_vector):
    """Return the four control points used by this module's cubic convention."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return np.array([
        start,
        start + np.asarray(start_vector, dtype=float),
        end + np.asarray(end_vector, dtype=float),
        end,
    ])


def split_cubic_bezier_controls(controls, parameter):
    """Split one cubic exactly at ``parameter`` using de Casteljau."""
    controls = np.asarray(controls, dtype=float)
    if controls.shape != (4, 2):
        raise ValueError("Cubic Bezier controls must have shape (4, 2)")
    if not 0.0 <= parameter <= 1.0:
        raise ValueError("Cubic Bezier split parameter must be in [0, 1]")
    level1 = (1.0 - parameter) * controls[:-1] + parameter * controls[1:]
    level2 = (1.0 - parameter) * level1[:-1] + parameter * level1[1:]
    point = (1.0 - parameter) * level2[0] + parameter * level2[1]
    return (
        np.array([controls[0], level1[0], level2[0], point]),
        np.array([point, level2[1], level1[2], controls[3]]),
    )


def subdivide_cubic_bezier_controls(controls, parameters):
    """Return exact cubic controls for consecutive parameter intervals."""
    parameters = np.asarray(parameters, dtype=float)
    if (
        parameters.ndim != 1
        or len(parameters) < 2
        or not np.isclose(parameters[0], 0.0)
        or not np.isclose(parameters[-1], 1.0)
        or np.any(np.diff(parameters) <= 0.0)
    ):
        raise ValueError(
            "Cubic subdivision parameters must increase strictly from 0 to 1"
        )
    remaining = np.asarray(controls, dtype=float)
    segments = []
    previous = 0.0
    for parameter in parameters[1:-1]:
        relative = (parameter - previous) / (1.0 - previous)
        segment, remaining = split_cubic_bezier_controls(remaining, relative)
        segments.append(segment)
        previous = parameter
    segments.append(remaining)
    return segments


def cubic_bezier_point_and_tangent_from_controls(controls, parameter):
    controls = np.asarray(controls, dtype=float)
    return cubic_bezier_point_and_tangent(
        controls[0], controls[3],
        controls[1] - controls[0], controls[2] - controls[3], parameter,
    )


def cubic_bezier_arc_parameters(controls, fractions, sample_count=1000):
    """Map arc-length fractions to actual parameters of one cubic."""
    controls = np.asarray(controls, dtype=float)
    parameters = np.linspace(0.0, 1.0, sample_count + 1)
    t = parameters[:, None]
    one_minus_t = 1.0 - t
    sampled_points = (
        one_minus_t ** 3 * controls[0]
        + 3.0 * one_minus_t ** 2 * t * controls[1]
        + 3.0 * one_minus_t * t ** 2 * controls[2]
        + t ** 3 * controls[3]
    )
    cumulative_lengths = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(
            np.diff(sampled_points, axis=0), axis=1
        )),
    ))
    if cumulative_lengths[-1] == 0.0:
        return np.asarray(fractions, dtype=float)
    return np.interp(
        fractions, cumulative_lengths / cumulative_lengths[-1], parameters
    )


def global_cubic_from_local_endpoint_controls(
    start, end, start_local, end_local, arc_fractions,
    sample_count=1000, max_iterations=80,
):
    """Fit one global cubic while retaining local endpoint control vectors.

    ``end_local`` is the last segment's inward control vector (P2 - P3).
    Arc-length matching and endpoint span scaling are a small coupled problem;
    fixed-point iteration retains the actual converged Bezier parameters.
    """
    fractions = np.asarray(arc_fractions, dtype=float)
    if (
        fractions.ndim != 1
        or len(fractions) < 2
        or not np.isclose(fractions[0], 0.0)
        or not np.isclose(fractions[-1], 1.0)
        or np.any(np.diff(fractions) <= 0.0)
    ):
        raise ValueError("Arc fractions must increase strictly from 0 to 1")
    parameters = np.array(fractions, copy=True)
    for unused_iteration in range(max_iterations):
        first_span = parameters[1]
        last_span = 1.0 - parameters[-2]
        controls = cubic_bezier_controls(
            start, end,
            np.asarray(start_local, dtype=float) / first_span,
            np.asarray(end_local, dtype=float) / last_span,
        )
        matched = cubic_bezier_arc_parameters(
            controls, fractions, sample_count=sample_count
        )
        if np.max(np.abs(matched - parameters)) < 2.e-11:
            parameters = matched
            break
        # Damping avoids a two-cycle for strongly curved endpoint data.
        parameters = 0.5 * (parameters + matched)
    controls = cubic_bezier_controls(
        start, end,
        np.asarray(start_local, dtype=float) / parameters[1],
        np.asarray(end_local, dtype=float) / (1.0 - parameters[-2]),
    )
    matched = cubic_bezier_arc_parameters(
        controls, fractions, sample_count=sample_count
    )
    if np.max(np.abs(matched - parameters)) > 2.e-7:
        raise ValueError("Global Bezier arc-parameter iteration did not converge")
    parameters = matched
    # One final rescale makes the endpoint identities exact for returned t_k.
    controls = cubic_bezier_controls(
        start, end,
        np.asarray(start_local, dtype=float) / parameters[1],
        np.asarray(end_local, dtype=float) / (1.0 - parameters[-2]),
    )
    points_and_tangents = [
        cubic_bezier_point_and_tangent_from_controls(controls, parameter)
        for parameter in parameters
    ]
    return {
        "controls": controls,
        "parameters": parameters,
        "points": [item[0] for item in points_and_tangents],
        "tangents": [item[1] for item in points_and_tangents],
        "segments": subdivide_cubic_bezier_controls(controls, parameters),
        "start_local": np.asarray(start_local, dtype=float),
        "end_local": np.asarray(end_local, dtype=float),
        "global_start_vector": controls[1] - controls[0],
        "global_end_vector": controls[2] - controls[3],
        "arc_fractions": np.array(fractions, copy=True),
    }


def global_cubic_data_from_controls(
    controls, arc_fractions, sample_count=1000,
):
    """Sample and exactly subdivide explicit global cubic controls."""
    controls = np.asarray(controls, dtype=float)
    parameters = cubic_bezier_arc_parameters(
        controls, arc_fractions, sample_count=sample_count
    )
    points_and_tangents = [
        cubic_bezier_point_and_tangent_from_controls(controls, parameter)
        for parameter in parameters
    ]
    return {
        "controls": np.array(controls, copy=True),
        "parameters": parameters,
        "points": [item[0] for item in points_and_tangents],
        "tangents": [item[1] for item in points_and_tangents],
        "segments": subdivide_cubic_bezier_controls(controls, parameters),
        "global_start_vector": controls[1] - controls[0],
        "global_end_vector": controls[2] - controls[3],
        "arc_fractions": np.asarray(arc_fractions, dtype=float),
    }


def cubic_bezier_surface_jacobian(control_net, u, v):
    """Return det(dX/du, dX/dv) for a 4 by 4 Bezier control net."""
    control_net = np.asarray(control_net, dtype=float)
    if control_net.shape != (2, 4, 4):
        raise ValueError("Bezier surface control net must have shape (2, 4, 4)")

    def quadratic_basis(value):
        return np.array([(1.0 - value) ** 2,
                         2.0 * value * (1.0 - value), value ** 2])

    def cubic_basis(value):
        return np.array([
            (1.0 - value) ** 3,
            3.0 * value * (1.0 - value) ** 2,
            3.0 * value ** 2 * (1.0 - value),
            value ** 3,
        ])

    du_controls = 3.0 * np.diff(control_net, axis=1)
    dv_controls = 3.0 * np.diff(control_net, axis=2)
    du = np.einsum(
        "i,xij,j->x", quadratic_basis(u), du_controls, cubic_basis(v)
    )
    dv = np.einsum(
        "i,xij,j->x", cubic_basis(u), dv_controls, quadratic_basis(v)
    )
    return float(du[0] * dv[1] - du[1] * dv[0])


def zero_mixed_bezier_control_net(
    bottom, top, left, right, mixed_controls=None,
):
    """Assemble the tensor-product net for four compatible cubic edges.

    ``mixed_controls`` follows corner order bottom-left, bottom-right,
    top-left, top-right.  Omitting it retains the new-node zero-mixed rule.
    """
    bottom = np.asarray(bottom, dtype=float)
    top = np.asarray(top, dtype=float)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if any(edge.shape != (4, 2) for edge in (bottom, top, left, right)):
        raise ValueError("Each candidate element edge must have four 2D controls")
    corners_match = (
        np.allclose(bottom[0], left[0])
        and np.allclose(bottom[3], right[0])
        and np.allclose(top[0], left[3])
        and np.allclose(top[3], right[3])
    )
    if not corners_match:
        raise ValueError("Candidate element edges do not share their corners")
    net = np.empty((2, 4, 4), dtype=float)
    net[:, :, 0] = bottom.T
    net[:, :, 3] = top.T
    net[:, 0, :] = left.T
    net[:, 3, :] = right.T
    # New patch nodes use a zero raw mixed derivative.  The four interior
    # controls are consequently position + the two adjacent controls.
    if mixed_controls is None:
        mixed_controls = [np.zeros(2) for unused_corner in range(4)]
    if len(mixed_controls) != 4:
        raise ValueError("Mixed controls must contain one vector per corner")
    for mixed, (i, j, corner_i, corner_j) in zip(mixed_controls, (
        (1, 1, 0, 0), (2, 1, 3, 0),
        (1, 2, 0, 3), (2, 2, 3, 3),
    )):
        corner = net[:, corner_i, corner_j]
        along = net[:, i, corner_j] - corner
        radial = net[:, corner_i, j] - corner
        net[:, i, j] = corner + along + radial + np.asarray(mixed)
    return net


def validate_bezier_control_nets(control_nets, sample_count=9):
    """Require a consistent, scale-aware nonzero Jacobian in every net."""
    if sample_count < 2:
        raise ValueError("Jacobian sampling needs at least two points per axis")
    minimum_signed = math.inf
    details = []
    for control_net in control_nets:
        control_net = np.asarray(control_net, dtype=float)
        span = np.ptp(control_net.reshape(2, -1), axis=1)
        characteristic_length = max(float(np.linalg.norm(span)), 1.e-12)
        tolerance = 1.e-8 * characteristic_length ** 2
        corner_orientation = np.cross(
            control_net[:, 3, 0] - control_net[:, 0, 0],
            control_net[:, 0, 3] - control_net[:, 0, 0],
        )
        orientation = float(np.sign(corner_orientation))
        if orientation == 0.0:
            orientation = 1.0
        signed_values = []
        for u in np.linspace(0.0, 1.0, sample_count):
            for v in np.linspace(0.0, 1.0, sample_count):
                signed_values.append(
                    orientation
                    * cubic_bezier_surface_jacobian(control_net, u, v)
                )
        local_minimum = min(signed_values)
        minimum_signed = min(minimum_signed, local_minimum)
        details.append((local_minimum, tolerance))
        if not np.isfinite(local_minimum) or local_minimum <= tolerance:
            return False, minimum_signed, details
    return True, minimum_signed, details


def sample_cubic_bezier_by_arc_fractions(
    start, end, start_vector, end_vector, fractions, sample_count=1000
):
    parameters = np.linspace(0., 1., sample_count + 1)
    sampled_points = np.array([
        cubic_bezier_point_and_tangent(
            start, end, start_vector, end_vector, parameter
        )[0]
        for parameter in parameters
    ])
    segment_lengths = np.linalg.norm(np.diff(sampled_points, axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.], np.cumsum(segment_lengths)))
    total_length = cumulative_lengths[-1]
    if total_length == 0:
        matched_parameters = np.asarray(fractions)
    else:
        matched_parameters = np.interp(
            fractions, cumulative_lengths / total_length, parameters
        )
    points = []
    tangents = []
    for parameter in matched_parameters:
        point, tangent = cubic_bezier_point_and_tangent(
            start, end, start_vector, end_vector, parameter
        )
        points.append(point)
        tangents.append(tangent)
    return points, tangents, matched_parameters


def element_by_index(element_index):
    return next(
        (element for element in element_list if element.index == element_index),
        None,
    )


def effective_node_basis_vector(node, basis_index, owner_edge=None):
    """Return one nodal basis as an effective scene-space control vector."""
    owner = (
        element_by_index(getattr(owner_edge, "element_index", None))
        if (
            owner_edge is not None
            and hasattr(owner_edge, "nodes")
            and hasattr(owner_edge, "sizes")
        ) else None
    )
    if owner is not None and hasattr(owner, "sizes"):
        local_vertices = [
            local_vertex
            for local_vertex, vertex in enumerate(owner.vertices)
            if vertex == node.index
        ]
        if len(local_vertices) == 1:
            scale = float(owner.sizes[basis_index, local_vertices[0]])
            return scale * np.asarray(node.xx[:, basis_index], dtype=float)
    return np.asarray(node_basis_display_vector(node, basis_index), dtype=float)


def cap_continuation_basis_vector(
    cap_node, new_neighbor, basis_index, owner_edge, direction_sign=1.0,
    continuation_direction=None,
):
    """Scale an old cap tangent to the adjacent new-patch chord length."""
    vector = effective_node_basis_vector(cap_node, basis_index, owner_edge)
    new_neighbor_position = (
        new_neighbor.position
        if hasattr(new_neighbor, "position") else new_neighbor
    )
    new_chord = np_point(new_neighbor_position) - np_point(cap_node.position)
    new_length = np.linalg.norm(new_chord)
    if new_length == 0.0:
        return vector
    direction = (
        direction_sign * new_chord
        if continuation_direction is None
        else np.asarray(continuation_direction, dtype=float)
    )
    if np.inner(vector, direction) < 0.0:
        vector = -vector

    owner = element_by_index(getattr(owner_edge, "element_index", None))
    if owner is None:
        return vector
    local_vertices = [
        local_vertex
        for local_vertex, vertex in enumerate(owner.vertices)
        if vertex == cap_node.index
    ]
    if len(local_vertices) != 1:
        return vector
    local_vertex = local_vertices[0]
    candidate_sides = {(local_vertex - 1) % 4, local_vertex}
    cap_side = getattr(owner_edge, "element_side", None)
    main_sides = candidate_sides - {cap_side}
    if len(main_sides) != 1:
        return vector
    side = main_sides.pop()
    other_local_vertex = (
        side if (side + 1) % 4 == local_vertex
        else (local_vertex + 1) % 4
    )
    old_node_index = int(owner.vertices[other_local_vertex])
    if not 0 <= old_node_index < len(node_list):
        return vector
    old_node = node_list[old_node_index]
    old_length = np.linalg.norm(
        np_point(cap_node.position) - np_point(old_node.position)
    )
    if old_length == 0.0:
        return vector
    return (new_length / old_length) * vector


def boundary_chain_outward_displacement(ordered_edges):
    outward_normals = []
    transverse_lengths = []
    for edge in ordered_edges:
        owner = element_by_index(edge.element_index)
        if owner is None:
            raise ValueError("Selected boundary edge has no owning element")
        edge_positions = [np_point(node.position) for node in edge.nodes]
        midpoint = (edge_positions[0] + edge_positions[1]) / 2.
        element_positions = [
            np_point(node_list[vertex].position) for vertex in owner.vertices
        ]
        centroid = np.mean(element_positions, axis=0)
        chord = edge_positions[1] - edge_positions[0]
        chord_length = np.linalg.norm(chord)
        if chord_length == 0:
            raise ValueError("Selected boundary edge has zero length")
        normal = np.array([-chord[1], chord[0]]) / chord_length
        if np.inner(normal, centroid - midpoint) > 0:
            normal = -normal
        outward_normals.append(normal)

        owner_vertices = list(owner.vertices)
        edge_indices = [
            owner_vertices.index(node.index) for node in edge.nodes
        ]
        vertex_count = len(owner.vertices)
        for edge_index, node in zip(edge_indices, edge.nodes):
            adjacent_indices = {
                (edge_index - 1) % vertex_count,
                (edge_index + 1) % vertex_count,
            } - set(edge_indices)
            for adjacent_index in adjacent_indices:
                transverse_lengths.append(np.linalg.norm(
                    np_point(node_list[owner.vertices[adjacent_index]].position)
                    - np_point(node.position)
                ))

    representative_normal = np.sum(outward_normals, axis=0)
    representative_norm = np.linalg.norm(representative_normal)
    if representative_norm == 0:
        representative_normal = outward_normals[0]
    else:
        representative_normal /= representative_norm
    valid_lengths = [length for length in transverse_lengths if length > 0]
    if valid_lengths:
        offset_magnitude = float(np.mean(valid_lengths))
    else:
        offset_magnitude = float(np.mean([
            np.linalg.norm(
                np_point(edge.nodes[1].position)
                - np_point(edge.nodes[0].position)
            )
            for edge in ordered_edges
        ]))
    return representative_normal * offset_magnitude


class extended_bezier_handle(QGraphicsEllipseItem):
    def __init__(self, patch, role, position, color):
        radius = EXTENDED_BEZIER_HANDLE_SIZE / 2.0
        super().__init__(
            -radius, -radius,
            EXTENDED_BEZIER_HANDLE_SIZE, EXTENDED_BEZIER_HANDLE_SIZE,
            patch,
        )
        self.patch = patch
        self.role = role
        self.setPos(position)
        self.setBrush(QBrush(color))
        self.setPen(graphics_handle_outline_pen())
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(Z_VECTOR_HANDLE)

    def move_to_scene(self, scene_position):
        if self.role == "cap_global_tangent":
            self.patch.move_one_cap_global_curve_handle(
                self, scene_position
            )
            return
        if self.role in ("global_start_tangent", "global_end_tangent"):
            self.patch.move_global_cap_tangent_handle(self, scene_position)
            return
        self.patch.automatic_outer_geometry = False
        new_position = self.patch.mapFromScene(scene_position)
        delta = new_position - self.pos()
        self.setPos(new_position)
        if self.role in ("start", "end"):
            tangent_role = self.role + "_tangent"
            tangent_handle = next(
                handle for handle in self.patch.bezier_handles
                if handle.role == tangent_role
            )
            tangent_handle.setPos(tangent_handle.pos() + delta)
        if self.patch.one_cap_global_curve_only:
            self.patch.update_one_cap_global_curve_from_handles()
        else:
            self.patch.update_bezier_from_handles()


class extended_patch(QGraphicsPathItem):
    def __init__(self, ordered_nodes, ordered_edges, can_commit=True):
        super().__init__()
        self.setZValue(Z_PATCH_PREVIEW)
        preview_pen = QPen(Qt.black)
        preview_pen.setWidthF(EXTENDED_PATCH_LINE_WIDTH)
        preview_pen.setCosmetic(True)
        self.setPen(preview_pen)
        self.ordered_nodes = ordered_nodes
        self.ordered_edges = ordered_edges
        self.outer_nodes = []
        self.radial_layers = 1
        self.preview_node_rows = [self.ordered_nodes]
        self.preview_along_vectors = []
        self.can_commit = can_commit
        self.capped_gap = None
        self.one_cap_topology = None
        self.bezier_mode = False
        self.bezier_handles = []
        self.outer_tangents = []
        self.outer_parameters = []
        self.one_cap_global_rows = {}
        self.one_cap_global_active = False
        self.one_cap_global_minimum_jacobian = None
        self._one_cap_global_warning_emitted = False
        self._suppress_one_cap_global_warning = False
        self._one_cap_baseline_outer = None
        self.global_cap_tangent_length = None
        self.global_cap_tangent_default_length = None
        self.global_cap_tangent_direction = None
        self.global_cap_tangent_last_valid_length = None
        self.one_cap_global_curve_only = False
        self.one_cap_global_curve_controls = None
        self.one_cap_global_cap_at_start = None
        self.one_cap_global_control_polygon = None
        self.one_cap_outer_arc_fractions = None
        self._one_cap_curve_diagnostic_printed = False
        self.base_radial_displacement = None
        self.automatic_outer_geometry = False
        self.redraw()

    @property
    def required_outer_node_count(self):
        return len(self.ordered_nodes)

    def rebuild_intermediate_preview_rows(self):
        if (
            self.bezier_mode
            and self.one_cap_topology is not None
            and self._one_cap_baseline_outer is not None
            and len(self.outer_nodes) == self.required_outer_node_count
        ):
            baseline = self._one_cap_baseline_outer
            for node, position in zip(self.outer_nodes, baseline["points"]):
                if isinstance(node, extended_patch_node):
                    node.position = qt_point(position)
            self.outer_tangents = [
                np.array(tangent, copy=True) for tangent in baseline["tangents"]
            ]
            self.outer_parameters = np.array(
                baseline["parameters"], copy=True
            )
        self.one_cap_global_rows = {}
        self.one_cap_global_active = False
        self.one_cap_global_minimum_jacobian = None
        defined_count = min(
            len(self.outer_nodes), self.required_outer_node_count
        )
        if defined_count == 0:
            rows = [self.ordered_nodes]
            topology = self.one_cap_topology or self.capped_gap
            if topology is not None:
                cap_nodes = (
                    getattr(topology, "start_cap_nodes", [])
                    or getattr(topology, "end_cap_nodes", [])
                )
                cap_at_start = bool(getattr(topology, "start_cap_nodes", []))
                for cap_node in cap_nodes[1:]:
                    row = [cap_node] if cap_at_start else [cap_node]
                    rows.append(row)
            self.preview_node_rows = rows
            self.preview_along_vectors = []
            return
        if defined_count == self.required_outer_node_count:
            preview_inner_nodes = self.ordered_nodes
            preview_outer_nodes = self.outer_nodes
        else:
            preview_inner_nodes = self.ordered_nodes[:defined_count]
            preview_outer_nodes = self.outer_nodes[:defined_count]
        rows = [preview_inner_nodes]
        for radial_index in range(1, self.radial_layers):
            fraction = radial_index / self.radial_layers
            rows.append([
                extended_patch_node(QPointF(
                    (1. - fraction) * inner_node.position.x()
                    + fraction * outer_node.position.x(),
                    (1. - fraction) * inner_node.position.y()
                    + fraction * outer_node.position.y(),
                ))
                for inner_node, outer_node in zip(
                    preview_inner_nodes, preview_outer_nodes
                )
            ])
        rows.append(preview_outer_nodes)
        topology = self.one_cap_topology or self.capped_gap
        if topology is not None:
            for row_index in range(1, min(len(rows), self.radial_layers + 1)):
                if getattr(topology, "start_cap_nodes", []):
                    rows[row_index][0] = topology.start_cap_nodes[row_index]
                if getattr(topology, "end_cap_nodes", []):
                    rows[row_index][-1] = topology.end_cap_nodes[row_index]
        self.preview_node_rows = rows
        self.rebuild_preview_along_vectors()
        if (
            self.bezier_mode
            and self.one_cap_topology is not None
            and not self.one_cap_global_curve_only
        ):
            self.rebuild_one_cap_global_rows()

    def outer_along_vectors(self):
        main_uv_index = self.main_uv_index()
        if self.bezier_mode:
            if (
                len(self.outer_tangents) != self.required_outer_node_count
                or len(self.outer_parameters) != self.required_outer_node_count
            ):
                raise ValueError("Bezier samples do not match the outer row")
            scales = bezier_nodal_parameter_scales(self.outer_parameters)
            vectors = [
                scales[index] * np.asarray(tangent, dtype=float) / 3.0
                for index, tangent in enumerate(self.outer_tangents)
            ]
        else:
            vectors = self.positional_along_vectors(self.outer_nodes)
        fixed_start = self.fixed_bezier_start_node()
        fixed_end = self.fixed_bezier_end_node()
        if fixed_start is not None and hasattr(fixed_start, "xx"):
            topology = self.one_cap_topology or self.capped_gap
            owner_edges = getattr(topology, "start_cap_edges", []) or []
            vector = cap_continuation_basis_vector(
                fixed_start, self.outer_nodes[1], main_uv_index,
                owner_edges[-1] if owner_edges else None,
            )
            vectors[0] = vector
        if fixed_end is not None and hasattr(fixed_end, "xx"):
            topology = self.one_cap_topology or self.capped_gap
            owner_edges = getattr(topology, "end_cap_edges", []) or []
            vector = cap_continuation_basis_vector(
                fixed_end, self.outer_nodes[-2], main_uv_index,
                owner_edges[-1] if owner_edges else None,
                direction_sign=-1.0,
            )
            vectors[-1] = vector
        return vectors

    @staticmethod
    def positional_along_vectors(nodes):
        positions = [np_point(node.position) for node in nodes]
        vectors = []
        for index in range(len(positions)):
            if index == 0:
                vector = (positions[1] - positions[0]) / 3.0
            elif index == len(positions) - 1:
                vector = (positions[-1] - positions[-2]) / 3.0
            else:
                vector = (positions[index + 1] - positions[index - 1]) / 6.0
            vectors.append(vector)
        return vectors

    def rebuild_preview_along_vectors(self):
        if (
            len(self.preview_node_rows) != self.radial_layers + 1
            or len(self.outer_nodes) != self.required_outer_node_count
            or not self.ordered_edges
            or not hasattr(self.ordered_edges[0], "uv_index")
        ):
            self.preview_along_vectors = []
            return
        main_uv_index = self.main_uv_index()
        positional_inner_vectors = self.positional_along_vectors(
            self.ordered_nodes
        )
        inner_vectors = []
        for index, node in enumerate(self.ordered_nodes):
            if hasattr(node, "xx"):
                owner_edge = self.ordered_edges[max(0, index - 1)]
                vector = effective_node_basis_vector(
                    node, main_uv_index, owner_edge
                )
                if np.inner(vector, positional_inner_vectors[index]) < 0.0:
                    vector = -vector
                inner_vectors.append(vector)
            else:
                inner_vectors.append(positional_inner_vectors[index])
        outer_vectors = self.outer_along_vectors()
        vector_rows = [inner_vectors]
        topology = self.one_cap_topology or self.capped_gap
        for radial_index in range(1, self.radial_layers):
            fraction = radial_index / self.radial_layers
            row_vectors = [
                (1.0 - fraction) * inner_vector + fraction * outer_vector
                for inner_vector, outer_vector in zip(
                    inner_vectors, outer_vectors
                )
            ]
            if topology is not None:
                if (
                    getattr(topology, "start_cap_nodes", [])
                    and hasattr(topology.start_cap_nodes[radial_index], "xx")
                ):
                    vector = cap_continuation_basis_vector(
                        topology.start_cap_nodes[radial_index],
                        self.preview_node_rows[radial_index][1], main_uv_index,
                        topology.start_cap_edges[radial_index - 1],
                    )
                    row_vectors[0] = vector
                if (
                    getattr(topology, "end_cap_nodes", [])
                    and hasattr(topology.end_cap_nodes[radial_index], "xx")
                ):
                    vector = cap_continuation_basis_vector(
                        topology.end_cap_nodes[radial_index],
                        self.preview_node_rows[radial_index][-2], main_uv_index,
                        topology.end_cap_edges[radial_index - 1],
                        direction_sign=-1.0,
                    )
                    row_vectors[-1] = vector
            vector_rows.append(row_vectors)
        vector_rows.append(outer_vectors)
        self.preview_along_vectors = vector_rows
        for row, vectors in zip(self.preview_node_rows[1:], vector_rows[1:]):
            for node, vector in zip(row, vectors):
                if isinstance(node, extended_patch_node):
                    node.along_vector = np.array(vector, copy=True)

    @staticmethod
    def _edge_controls_in_order(edge, start_node, end_node):
        controls = edge_bezier_points(edge.points)
        controls = np.array([
            controls[:, 0, 0], controls[:, 1, 0],
            controls[:, 1, 1], controls[:, 0, 1],
        ])
        if edge.nodes[0] is start_node and edge.nodes[1] is end_node:
            return controls
        if edge.nodes[1] is start_node and edge.nodes[0] is end_node:
            return controls[::-1]
        raise ValueError("Bezier edge endpoints do not match the requested order")

    def _one_cap_candidate_control_nets(self, cap_first_rows_data):
        """Build candidate cell nets in cap-to-free, inner-to-outer order."""
        topology = self.one_cap_topology
        (
            inner_nodes, inner_edges, cap_rows, cap_nodes, cap_edges,
        ) = cap_first_rows(topology, self.preview_node_rows)
        row_segments = {
            0: [
                self._edge_controls_in_order(
                    next(
                        edge for edge in inner_edges
                        if frozenset(edge.vertices) == frozenset((
                            inner_nodes[index].index,
                            inner_nodes[index + 1].index,
                        ))
                    ),
                    inner_nodes[index], inner_nodes[index + 1]
                )
                for index in range(len(inner_edges))
            ]
        }
        row_segments.update({
            radial_index: data["segments"]
            for radial_index, data in cap_first_rows_data.items()
        })
        row_points = {
            0: [np_point(node.position) for node in inner_nodes]
        }
        row_points.update({
            radial_index: data["points"]
            for radial_index, data in cap_first_rows_data.items()
        })
        control_nets = []
        main_uv_index = self.main_uv_index()
        radial_uv_index = main_uv_index % 2 + 1

        def radial_controls(radial_index, column):
            bottom = np.asarray(row_points[radial_index][column])
            top = np.asarray(row_points[radial_index + 1][column])
            if column == 0:
                return self._edge_controls_in_order(
                    cap_edges[radial_index],
                    cap_nodes[radial_index], cap_nodes[radial_index + 1],
                )
            if radial_index == 0:
                node = inner_nodes[column]
                owner_edge = next(
                    edge for edge in inner_edges
                    if node.index in edge.vertices
                    and (
                        column == len(inner_nodes) - 1
                        or inner_nodes[column - 1].index in edge.vertices
                    )
                )
                radial_size = inherited_transverse_endpoint_size(
                    owner_edge, node, qt_point(top)
                )
                outward = radial_size * np.asarray(
                    node.xx[:, radial_uv_index], dtype=float
                )
            else:
                previous = np.asarray(row_points[radial_index - 1][column])
                outward = (bottom - previous) / 3.0
            inward = (bottom - top) / 3.0
            return np.array([bottom, bottom + outward, top + inward, top])

        def effective_mixed(node, along, radial):
            if not hasattr(node, "xx"):
                return np.zeros(2)
            raw_mixed = np.asarray(node.xx[:, 3], dtype=float)
            if np.linalg.norm(raw_mixed) == 0.0:
                return np.zeros(2)
            along_size = basis_size_for_effective_vector(
                node.xx[:, main_uv_index], along
            )
            radial_size = basis_size_for_effective_vector(
                node.xx[:, radial_uv_index], radial
            )
            return along_size * radial_size * raw_mixed

        for radial_index in range(self.radial_layers):
            bottom_points = row_points[radial_index]
            top_points = row_points[radial_index + 1]
            for column in range(len(inner_edges)):
                bottom = row_segments[radial_index][column]
                top = row_segments[radial_index + 1][column]
                left = radial_controls(radial_index, column)
                right = radial_controls(radial_index, column + 1)
                bottom_nodes = cap_rows[radial_index]
                top_nodes = cap_rows[radial_index + 1]
                mixed_controls = [
                    effective_mixed(
                        bottom_nodes[column],
                        bottom[1] - bottom[0], left[1] - left[0],
                    ),
                    effective_mixed(
                        bottom_nodes[column + 1],
                        bottom[2] - bottom[3], right[1] - right[0],
                    ),
                    effective_mixed(
                        top_nodes[column],
                        top[1] - top[0], left[2] - left[3],
                    ),
                    effective_mixed(
                        top_nodes[column + 1],
                        top[2] - top[3], right[2] - right[3],
                    ),
                ]
                control_nets.append(zero_mixed_bezier_control_net(
                    bottom, top, left, right,
                    mixed_controls=mixed_controls,
                ))
        return control_nets

    def rebuild_one_cap_global_rows(self):
        """Replace every new one-cap row by one exactly subdivided cubic."""
        if (
            len(self.preview_node_rows) != self.radial_layers + 1
            or len(self.preview_along_vectors) != self.radial_layers + 1
            or len(self.outer_nodes) != self.required_outer_node_count
        ):
            return
        topology = self.one_cap_topology
        if not all(
            hasattr(topology, attribute)
            for attribute in ("inner_nodes", "inner_edges")
        ):
            return
        start_cap_edges = list(getattr(topology, "start_cap_edges", []) or [])
        end_cap_edges = list(getattr(topology, "end_cap_edges", []) or [])
        if bool(start_cap_edges) == bool(end_cap_edges):
            return
        if any(
            not hasattr(edge, "points")
            for edge in list(getattr(topology, "inner_edges", []))
            + start_cap_edges + end_cap_edges
        ):
            return
        cap_at_start = bool(start_cap_edges)
        cap_edges = list(
            start_cap_edges if cap_at_start else end_cap_edges
        )
        cap_first_data = {}
        try:
            for radial_index in range(1, self.radial_layers + 1):
                baseline_row = self.preview_node_rows[radial_index]
                baseline_vectors = self.preview_along_vectors[radial_index]
                if cap_at_start:
                    row = list(baseline_row)
                    vectors = [np.asarray(vector) for vector in baseline_vectors]
                else:
                    row = list(reversed(baseline_row))
                    vectors = [
                        -np.asarray(vector)
                        for vector in reversed(baseline_vectors)
                    ]
                fractions = cumulative_node_length_fractions(row)
                if np.any(np.diff(fractions) <= 0.0):
                    raise ValueError("One-cap baseline row has repeated nodes")
                start = np_point(row[0].position)
                end = np_point(row[-1].position)
                start_local = cap_continuation_basis_vector(
                    row[0], row[1], self.main_uv_index(),
                    cap_edges[radial_index - 1],
                    continuation_direction=end - start,
                )
                last_chord = np_point(row[-1].position - row[-2].position)
                end_basis = np.asarray(vectors[-1], dtype=float)
                end_sign = np.sign(np.inner(end_basis, -last_chord)) or -1.0
                end_local = end_sign * end_basis
                data = global_cubic_from_local_endpoint_controls(
                    start, end, start_local, end_local, fractions
                )
                cap_first_data[radial_index] = data

            default_outer = cap_first_data[self.radial_layers]
            default_outer_vector = np.asarray(
                default_outer["global_start_vector"], dtype=float
            )
            default_length = np.linalg.norm(default_outer_vector)
            if not np.isfinite(default_length) or default_length <= 0.0:
                raise ValueError("One-cap global tangent length is zero")
            if self.global_cap_tangent_default_length is None:
                self.global_cap_tangent_default_length = default_length
            if self.global_cap_tangent_direction is None:
                self.global_cap_tangent_direction = (
                    default_outer_vector / default_length
                )
            if self.global_cap_tangent_length is not None:
                scale = self.global_cap_tangent_length / default_length
                adjusted_data = {}
                for radial_index, default_data in cap_first_data.items():
                    controls = np.array(default_data["controls"], copy=True)
                    controls[1] = controls[0] + scale * (
                        controls[1] - controls[0]
                    )
                    data = global_cubic_data_from_controls(
                        controls, default_data["arc_fractions"]
                    )
                    data["start_local"] = (
                        data["parameters"][1]
                        * data["global_start_vector"]
                    )
                    data["end_local"] = (
                        (1.0 - data["parameters"][-2])
                        * data["global_end_vector"]
                    )
                    adjusted_data[radial_index] = data
                cap_first_data = adjusted_data

            control_nets = self._one_cap_candidate_control_nets(cap_first_data)
            valid, minimum_jacobian, unused_details = (
                validate_bezier_control_nets(control_nets, sample_count=9)
            )
            if not valid:
                raise ValueError(
                    "candidate Jacobian {:.6g} is non-positive or too small".format(
                        minimum_jacobian
                    )
                )
        except (ValueError, FloatingPointError) as error:
            if (
                not self._suppress_one_cap_global_warning
                and not self._one_cap_global_warning_emitted
            ):
                print(
                    "One-cap global Bezier rejected; using previous geometry:",
                    error,
                )
                self._one_cap_global_warning_emitted = True
            return

        original_data = {}
        for radial_index, data in cap_first_data.items():
            if cap_at_start:
                converted = dict(data)
            else:
                converted = dict(data)
                converted["parameters"] = 1.0 - data["parameters"][::-1]
                converted["points"] = list(reversed(data["points"]))
                converted["tangents"] = [
                    -tangent for tangent in reversed(data["tangents"])
                ]
                converted["segments"] = [
                    segment[::-1] for segment in reversed(data["segments"])
                ]
                converted["start_local"] = np.array(
                    data["end_local"], copy=True
                )
                converted["end_local"] = np.array(
                    data["start_local"], copy=True
                )
                converted["global_start_vector"] = (
                    converted["controls"][2] - converted["controls"][3]
                )
                converted["global_end_vector"] = (
                    converted["controls"][1] - converted["controls"][0]
                )
                converted["controls"] = data["controls"][::-1]
            original_data[radial_index] = converted
            for node, position in zip(
                self.preview_node_rows[radial_index], converted["points"]
            ):
                if isinstance(node, extended_patch_node):
                    node.position = qt_point(position)
            scales = bezier_nodal_parameter_scales(converted["parameters"])
            raw_vectors = [
                scales[index] * np.asarray(tangent) / 3.0
                for index, tangent in enumerate(converted["tangents"])
            ]
            # A fixed cap node keeps its existing raw basis.  Expose the exact
            # effective first-segment control in the preview vector array.
            cap_column = 0 if cap_at_start else -1
            raw_vectors[cap_column] = np.array(
                data["start_local"] if cap_at_start else -data["start_local"],
                copy=True,
            )
            self.preview_along_vectors[radial_index] = raw_vectors
            for node, vector in zip(
                self.preview_node_rows[radial_index], raw_vectors
            ):
                if isinstance(node, extended_patch_node):
                    node.along_vector = np.array(vector, copy=True)

        self.one_cap_global_rows = original_data
        self.one_cap_global_active = True
        self._one_cap_global_warning_emitted = False
        self.one_cap_global_minimum_jacobian = minimum_jacobian
        outer_data = original_data[self.radial_layers]
        self.outer_parameters = np.array(outer_data["parameters"], copy=True)
        self.outer_tangents = [
            np.array(tangent, copy=True) for tangent in outer_data["tangents"]
        ]

    def set_radial_layers(self, nr):
        if not 1 <= nr <= MAX_RADIAL_LAYERS:
            raise ValueError(
                "Radial layer count must be between 1 and {}".format(
                    MAX_RADIAL_LAYERS
                )
            )
        topology = self.one_cap_topology or self.capped_gap
        if topology is not None and (
            getattr(topology, "start_cap_edges", [])
            or getattr(topology, "end_cap_edges", [])
        ):
            print("Radial layer count is fixed by the selected cap chain")
            return
        self.radial_layers = nr
        if self.automatic_outer_geometry:
            self.regenerate_automatic_outer_geometry()
        else:
            self.rebuild_intermediate_preview_rows()
            self.redraw(rebuild_rows=False)

    def one_layer_radial_displacement(self):
        if self.base_radial_displacement is not None:
            return np.array(self.base_radial_displacement, copy=True)
        try:
            displacement = boundary_chain_outward_displacement(
                self.ordered_edges
            )
        except (AttributeError, ValueError, NameError):
            topology = self.one_cap_topology or self.capped_gap
            fixed_node = None
            inner_node = None
            if topology is not None:
                if topology.outer_start_node is not None:
                    fixed_node = topology.outer_start_node
                    inner_node = self.ordered_nodes[0]
                elif topology.outer_end_node is not None:
                    fixed_node = topology.outer_end_node
                    inner_node = self.ordered_nodes[-1]
            if fixed_node is not None:
                displacement = (
                    np_point(fixed_node.position) - np_point(inner_node.position)
                ) / self.radial_layers
            else:
                inner_start = self.ordered_nodes[0].position
                inner_end = self.ordered_nodes[-1].position
                chord = inner_end - inner_start
                chord_length = math.hypot(chord.x(), chord.y())
                if chord_length == 0:
                    displacement = np.array([0., -50.])
                else:
                    displacement = 50. * np.array([
                        -chord.y() / chord_length,
                        chord.x() / chord_length,
                    ])
        self.base_radial_displacement = np.array(displacement, copy=True)
        return np.array(self.base_radial_displacement, copy=True)

    def automatic_outer_endpoints(self):
        topology = self.one_cap_topology or self.capped_gap
        displacement = qt_point(
            self.radial_layers * self.one_layer_radial_displacement()
        )
        start = self.ordered_nodes[0].position + displacement
        end = self.ordered_nodes[-1].position + displacement
        if topology is not None:
            if topology.outer_start_node is not None:
                start = topology.outer_start_node.position
            if topology.outer_end_node is not None:
                end = topology.outer_end_node.position
        return start, end

    def initialize_automatic_outer_geometry(self):
        topology = self.one_cap_topology or self.capped_gap
        if (
            topology is not None
            and topology.outer_start_node is not None
            and topology.outer_end_node is not None
        ):
            return
        start, end = self.automatic_outer_endpoints()
        if topology is None:
            displacement = qt_point(
                self.radial_layers * self.one_layer_radial_displacement()
            )
            positions = [
                node.position + displacement for node in self.ordered_nodes
            ]
        else:
            positions = interpolate_positions(
                start, end, len(self.ordered_edges)
            )
        self.set_outer_positions(positions, automatic=True)
        if topology is not None:
            if topology.outer_start_node is not None:
                self.outer_nodes[0] = topology.outer_start_node
            if topology.outer_end_node is not None:
                self.outer_nodes[-1] = topology.outer_end_node
            self.redraw()

    def regenerate_automatic_outer_geometry(self):
        if self.bezier_mode:
            start, end = self.automatic_outer_endpoints()
            handles = {handle.role: handle for handle in self.bezier_handles}
            chord = end - start
            if "start" in handles:
                handles["start"].setPos(start)
            if "end" in handles:
                handles["end"].setPos(end)
            if "start_tangent" in handles:
                handles["start_tangent"].setPos(start + chord / 3.)
            if "end_tangent" in handles:
                handles["end_tangent"].setPos(end - chord / 3.)
            self.update_bezier_from_handles()
        else:
            self.initialize_automatic_outer_geometry()

    def add_outer_node(self, position):
        if self.automatic_outer_geometry:
            self.outer_nodes = []
            self.automatic_outer_geometry = False
        if self.one_cap_topology is not None:
            if self.outer_nodes:
                return False
            topology = self.one_cap_topology
            start = (
                topology.outer_start_node.position
                if topology.outer_start_node is not None else position
            )
            end = (
                topology.outer_end_node.position
                if topology.outer_end_node is not None else position
            )
            segment_count = len(self.ordered_edges)
            self.set_outer_positions(interpolate_positions(
                start, end, segment_count
            ))
            return True
        if len(self.outer_nodes) >= self.required_outer_node_count:
            return False
        self.outer_nodes.append(extended_patch_node(position))
        if len(self.outer_nodes) == self.required_outer_node_count:
            self.outer_nodes = order_outer_nodes(
                self.ordered_nodes, self.outer_nodes
            )
        self.redraw()
        return True

    def set_outer_positions(self, positions, automatic=False):
        if len(positions) != self.required_outer_node_count:
            raise ValueError("Outer position count does not match inner nodes")
        self.outer_nodes = [
            extended_patch_node(position) for position in positions
        ]
        self.automatic_outer_geometry = automatic
        if self.one_cap_global_curve_only:
            fixed_start = self.fixed_bezier_start_node()
            fixed_end = self.fixed_bezier_end_node()
            if fixed_start is not None:
                self.outer_nodes[0] = fixed_start
                endpoint_role = "end"
                free_position = self.outer_nodes[-1].position
            else:
                self.outer_nodes[-1] = fixed_end
                endpoint_role = "start"
                free_position = self.outer_nodes[0].position
            handles = {handle.role: handle for handle in self.bezier_handles}
            endpoint_handle = handles[endpoint_role]
            tangent_handle = handles[endpoint_role + "_tangent"]
            delta = free_position - endpoint_handle.pos()
            endpoint_handle.setPos(free_position)
            tangent_handle.setPos(tangent_handle.pos() + delta)
            self.update_one_cap_global_curve_from_handles()
            return
        self.redraw()

    def initial_one_cap_global_curve_controls(self, start, end):
        """Initialize one global cubic independently of mesh subdivision."""
        topology = self.one_cap_topology
        cap_at_start = topology.outer_start_node is not None
        start_array = np_point(start)
        end_array = np_point(end)
        chord = end_array - start_array
        chord_length = np.linalg.norm(chord)
        if not np.isfinite(chord_length) or chord_length == 0.0:
            raise ValueError("One-cap global curve has zero chord length")
        handles = {handle.role: handle for handle in self.bezier_handles}
        if cap_at_start:
            cap_edges = topology.start_cap_edges
            fixed_node = topology.outer_start_node
            free_control = np_point(handles["end_tangent"].pos())
            continuation = chord
            free_endpoint = end_array
        else:
            cap_edges = topology.end_cap_edges
            fixed_node = topology.outer_end_node
            free_control = np_point(handles["start_tangent"].pos())
            continuation = -chord
            free_endpoint = start_array
        inherited = cap_continuation_basis_vector(
            fixed_node, qt_point(free_endpoint), self.main_uv_index(),
            cap_edges[-1] if cap_edges else None,
            continuation_direction=continuation,
        )
        inherited_norm = np.linalg.norm(inherited)
        if not np.isfinite(inherited_norm) or inherited_norm == 0.0:
            inherited = continuation
            inherited_norm = chord_length
        cap_vector = chord_length * inherited / (3.0 * inherited_norm)
        if cap_at_start:
            return np.array([
                start_array, start_array + cap_vector,
                free_control, end_array,
            ])
        return np.array([
            start_array, free_control, end_array + cap_vector, end_array,
        ])

    def enable_one_cap_global_curve_only(self, start, end, has_start_cap):
        """Expose an isolated editable 1D global outer-boundary cubic."""
        controls = self.initial_one_cap_global_curve_controls(start, end)
        self.one_cap_global_curve_only = True
        self.one_cap_global_cap_at_start = bool(has_start_cap)
        self.one_cap_global_curve_controls = controls
        if has_start_cap:
            cap_vector = controls[1] - controls[0]
            free_positions = {
                "end": controls[3], "end_tangent": controls[2]
            }
            fixed_position = self.bezier_start_position()
        else:
            cap_vector = controls[2] - controls[3]
            free_positions = {
                "start": controls[0], "start_tangent": controls[1]
            }
            fixed_position = self.bezier_end_position()
        for handle in self.bezier_handles:
            if handle.role in free_positions:
                handle.setPos(qt_point(free_positions[handle.role]))
        length = np.linalg.norm(cap_vector)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("One-cap global tangent length is zero")
        self.global_cap_tangent_length = length
        self.global_cap_tangent_default_length = length
        self.global_cap_tangent_last_valid_length = length
        self.global_cap_tangent_direction = cap_vector / length
        self.bezier_handles.append(extended_bezier_handle(
            self, "cap_global_tangent",
            fixed_position + qt_point(cap_vector), QColor(255, 128, 0),
        ))
        control_pen = QPen(QColor(96, 96, 96))
        control_pen.setWidthF(0.9)
        control_pen.setCosmetic(True)
        self.one_cap_global_control_polygon = QGraphicsPathItem(self)
        self.one_cap_global_control_polygon.setPen(control_pen)
        self.one_cap_global_control_polygon.setZValue(0.25)
        self.update_one_cap_global_curve_from_handles()
        self.print_one_cap_global_curve_diagnostic()

    def enable_bezier_mode(self, one_cap_curve_only=False):
        if self.bezier_mode:
            return
        topology = self.one_cap_topology or self.capped_gap
        has_start_cap = (
            topology is not None and topology.outer_start_node is not None
        )
        has_end_cap = (
            topology is not None and topology.outer_end_node is not None
        )
        has_complete_outer_row = (
            len(self.outer_nodes) == self.required_outer_node_count
        )
        if has_start_cap and has_end_cap:
            start = self.outer_nodes[0].position
            end = self.outer_nodes[-1].position
        elif has_complete_outer_row:
            start = self.outer_nodes[0].position
            end = self.outer_nodes[-1].position
        elif has_start_cap or has_end_cap:
            start, end = self.automatic_outer_endpoints()
            self.automatic_outer_geometry = True
        else:
            start, end = self.automatic_outer_endpoints()
            self.automatic_outer_geometry = True
        chord = end - start
        start_vector = chord / 3.
        end_vector = -chord / 3.
        self.bezier_handles = []
        if not has_start_cap:
            self.bezier_handles.append(extended_bezier_handle(
                self, "start", start, QColor(255, 255, 0)
            ))
        if not has_end_cap:
            self.bezier_handles.append(extended_bezier_handle(
                self, "end", end, QColor(255, 255, 0)
            ))
        if not has_start_cap:
            self.bezier_handles.append(extended_bezier_handle(
                self, "start_tangent", start + start_vector,
                QColor(0, 255, 255),
            ))
        if not has_end_cap:
            self.bezier_handles.append(extended_bezier_handle(
                self, "end_tangent", end + end_vector,
                QColor(0, 255, 255),
            ))
        self.bezier_mode = True
        if one_cap_curve_only and self.one_cap_topology is not None:
            self.enable_one_cap_global_curve_only(
                start, end, has_start_cap
            )
            return
        self.update_bezier_from_handles()
        if self.one_cap_topology is not None:
            role = (
                "global_start_tangent"
                if has_start_cap else "global_end_tangent"
            )
            fixed_position = (
                self.bezier_start_position()
                if has_start_cap else self.bezier_end_position()
            )
            if self.one_cap_global_active:
                vector = (
                    self.bezier_start_vector()
                    if has_start_cap else self.bezier_end_vector()
                )
            elif (
                self.global_cap_tangent_direction is not None
                and self.global_cap_tangent_default_length is not None
            ):
                vector = (
                    self.global_cap_tangent_default_length
                    * np.asarray(
                        self.global_cap_tangent_direction, dtype=float
                    )
                )
            else:
                vector = np.zeros(2)
            length = np.linalg.norm(vector)
            if np.isfinite(length) and length > 0.0:
                self.global_cap_tangent_length = length
                self.global_cap_tangent_last_valid_length = (
                    length if self.one_cap_global_active else None
                )
                self.global_cap_tangent_direction = vector / length
                self.bezier_handles.append(extended_bezier_handle(
                    self, role, fixed_position + qt_point(vector),
                    QColor(255, 128, 0),
                ))
                self.redraw(rebuild_rows=False)

    def global_cap_tangent_handle(self):
        return next((
            handle for handle in self.bezier_handles
            if handle.role in (
                "cap_global_tangent", "global_start_tangent",
                "global_end_tangent",
            )
        ), None)

    def update_one_cap_global_curve_from_handles(self):
        """Update authoritative controls and rebuild their patch preview."""
        if not self.one_cap_global_curve_only:
            return
        handles = {handle.role: handle for handle in self.bezier_handles}
        direction = np.asarray(
            self.global_cap_tangent_direction, dtype=float
        )
        cap_vector = self.global_cap_tangent_length * direction
        if self.one_cap_global_cap_at_start:
            fixed = np_point(self.bezier_start_position())
            controls = np.array([
                fixed,
                fixed + cap_vector,
                np_point(handles["end_tangent"].pos()),
                np_point(handles["end"].pos()),
            ])
        else:
            fixed = np_point(self.bezier_end_position())
            controls = np.array([
                np_point(handles["start"].pos()),
                np_point(handles["start_tangent"].pos()),
                fixed + cap_vector,
                fixed,
            ])
        self.one_cap_global_curve_controls = controls
        handles["cap_global_tangent"].setPos(
            qt_point(fixed + cap_vector)
        )
        self.resample_one_cap_global_outer_curve()
        self.redraw(rebuild_rows=False)

    def resample_one_cap_global_outer_curve(self):
        """Sample authoritative controls without feeding samples back to them."""
        controls = np.asarray(self.one_cap_global_curve_controls, dtype=float)
        controls_before = np.array(controls, copy=True)
        fractions = cumulative_node_length_fractions(self.ordered_nodes)
        points, tangents, parameters = sample_cubic_bezier_by_arc_fractions(
            controls[0], controls[3],
            controls[1] - controls[0], controls[2] - controls[3],
            fractions,
        )
        sampled_nodes = [
            extended_patch_node(qt_point(point)) for point in points
        ]
        if self.one_cap_global_cap_at_start:
            sampled_nodes[0] = self.one_cap_topology.outer_start_node
        else:
            sampled_nodes[-1] = self.one_cap_topology.outer_end_node
        self.outer_nodes = sampled_nodes
        self.outer_tangents = [
            np.array(tangent, copy=True) for tangent in tangents
        ]
        self.outer_parameters = np.array(parameters, copy=True)
        self.one_cap_outer_arc_fractions = np.array(fractions, copy=True)
        self._one_cap_baseline_outer = None
        self.rebuild_intermediate_preview_rows()
        if not np.array_equal(
            self.one_cap_global_curve_controls, controls_before
        ):
            raise RuntimeError("Patch preview modified global curve controls")

    def move_one_cap_global_curve_handle(self, handle, scene_position):
        """Slide the isolated curve's cap control along its fixed tangent."""
        if (
            not self.one_cap_global_curve_only
            or handle is not self.global_cap_tangent_handle()
        ):
            raise ValueError("Cap tangent handle does not belong to curve")
        direction = np.asarray(
            self.global_cap_tangent_direction, dtype=float
        )
        direction /= np.linalg.norm(direction)
        fixed_position = (
            self.bezier_start_position()
            if self.one_cap_global_cap_at_start
            else self.bezier_end_position()
        )
        fixed = np_point(fixed_position)
        mouse = np_point(self.mapFromScene(scene_position))
        controls = np.asarray(self.one_cap_global_curve_controls)
        geometry_scale = max(
            np.linalg.norm(controls[3] - controls[0]),
            float(self.global_cap_tangent_default_length), 1.e-12,
        )
        minimum_length = 1.e-6 * geometry_scale
        self.global_cap_tangent_length = max(
            float(np.inner(mouse - fixed, direction)), minimum_length
        )
        self.global_cap_tangent_last_valid_length = (
            self.global_cap_tangent_length
        )
        self.automatic_outer_geometry = False
        self.update_one_cap_global_curve_from_handles()
        return True

    def print_one_cap_global_curve_diagnostic(self):
        if self._one_cap_curve_diagnostic_printed:
            return
        controls = np.asarray(self.one_cap_global_curve_controls)
        print("One-cap global 1D Bezier controls:")
        for index, control in enumerate(controls):
            print("  P{} = {}".format(index, control))
        print("  cap tangent unit =", self.global_cap_tangent_direction)
        print("  lambda_cap =", self.global_cap_tangent_length)
        for handle in self.bezier_handles:
            position = handle.pos()
            scene_position = handle.scenePos()
            print(
                "  handle {} pos=({}, {}) scenePos=({}, {}) "
                "visible={} enabled={}".format(
                    handle.role, position.x(), position.y(),
                    scene_position.x(), scene_position.y(),
                    handle.isVisible(), handle.isEnabled(),
                )
            )
        self._one_cap_curve_diagnostic_printed = True

    def sync_global_cap_tangent_handle(self):
        handle = self.global_cap_tangent_handle()
        if handle is None or not self.one_cap_global_active:
            return
        if handle.role == "global_start_tangent":
            fixed_position = self.bezier_start_position()
            vector = self.bezier_start_vector()
        else:
            fixed_position = self.bezier_end_position()
            vector = self.bezier_end_vector()
        handle.setPos(fixed_position + qt_point(vector))

    def set_interactive_patch_status(self, message):
        patch_scene = self.scene()
        if patch_scene is None:
            return
        for patch_view in patch_scene.views():
            if (
                getattr(patch_view, "current_extended_patch", None) is self
                and hasattr(patch_view, "set_patch_status")
            ):
                patch_view.set_patch_status(message)

    def move_global_cap_tangent_handle(self, handle, scene_position):
        """Project a cap-side global handle drag onto its inherited tangent."""
        if handle is not self.global_cap_tangent_handle():
            raise ValueError("Global cap tangent handle does not belong to patch")
        direction = np.asarray(
            self.global_cap_tangent_direction, dtype=float
        )
        direction_norm = np.linalg.norm(direction)
        if not np.isfinite(direction_norm) or direction_norm == 0.0:
            return False
        direction = direction / direction_norm
        fixed_position = (
            self.bezier_start_position()
            if handle.role == "global_start_tangent"
            else self.bezier_end_position()
        )
        fixed = np_point(fixed_position)
        mouse = np_point(self.mapFromScene(scene_position))
        chord_length = np.linalg.norm(
            np_point(self.bezier_end_position())
            - np_point(self.bezier_start_position())
        )
        geometry_scale = max(
            chord_length,
            float(self.global_cap_tangent_default_length or 0.0),
            1.e-12,
        )
        minimum_length = 1.e-6 * geometry_scale
        proposed_length = max(
            float(np.inner(mouse - fixed, direction)), minimum_length
        )
        previous_length = self.global_cap_tangent_length
        self.global_cap_tangent_length = proposed_length
        self.automatic_outer_geometry = False
        self._suppress_one_cap_global_warning = True
        try:
            self.update_bezier_from_handles()
            if not self.one_cap_global_active:
                self.global_cap_tangent_length = previous_length
                self.update_bezier_from_handles()
                self.sync_global_cap_tangent_handle()
                self.set_interactive_patch_status("Invalid patch geometry")
                return False
            self.global_cap_tangent_last_valid_length = proposed_length
            self.sync_global_cap_tangent_handle()
            return True
        finally:
            self._suppress_one_cap_global_warning = False

    def fixed_bezier_start_node(self):
        topology = self.one_cap_topology or self.capped_gap
        if topology is None:
            return None
        return topology.outer_start_node

    def fixed_bezier_end_node(self):
        topology = self.one_cap_topology or self.capped_gap
        if topology is None:
            return None
        return topology.outer_end_node

    def bezier_start_position(self):
        fixed_node = self.fixed_bezier_start_node()
        if fixed_node is not None:
            return fixed_node.position
        return next(
            handle.pos() for handle in self.bezier_handles
            if handle.role == "start"
        )

    def bezier_end_position(self):
        fixed_node = self.fixed_bezier_end_node()
        if fixed_node is not None:
            return fixed_node.position
        return next(
            handle.pos() for handle in self.bezier_handles
            if handle.role == "end"
        )

    def main_uv_index(self):
        return self.ordered_edges[0].uv_index

    def bezier_start_vector(self):
        if self.one_cap_global_active and self.radial_layers in self.one_cap_global_rows:
            controls = self.one_cap_global_rows[self.radial_layers]["controls"]
            return np.asarray(controls[1] - controls[0], dtype=float)
        fixed_node = self.fixed_bezier_start_node()
        start = np_point(self.bezier_start_position())
        end = np_point(self.bezier_end_position())
        if fixed_node is not None:
            topology = self.one_cap_topology or self.capped_gap
            owner_edges = getattr(topology, "start_cap_edges", []) or []
            new_neighbor = (
                self.outer_nodes[1]
                if len(self.outer_nodes) == self.required_outer_node_count
                else qt_point(
                    start + (end - start) / len(self.ordered_edges)
                )
            )
            vector = cap_continuation_basis_vector(
                fixed_node, new_neighbor, self.main_uv_index(),
                owner_edges[-1] if owner_edges else None,
                continuation_direction=end - start,
            )
            return vector
        tangent_position = next(
            handle.pos() for handle in self.bezier_handles
            if handle.role == "start_tangent"
        )
        return np_point(tangent_position - self.bezier_start_position())

    def bezier_end_vector(self):
        if self.one_cap_global_active and self.radial_layers in self.one_cap_global_rows:
            controls = self.one_cap_global_rows[self.radial_layers]["controls"]
            return np.asarray(controls[2] - controls[3], dtype=float)
        fixed_node = self.fixed_bezier_end_node()
        start = np_point(self.bezier_start_position())
        end = np_point(self.bezier_end_position())
        if fixed_node is not None:
            topology = self.one_cap_topology or self.capped_gap
            owner_edges = getattr(topology, "end_cap_edges", []) or []
            new_neighbor = (
                self.outer_nodes[-2]
                if len(self.outer_nodes) == self.required_outer_node_count
                else qt_point(
                    end - (end - start) / len(self.ordered_edges)
                )
            )
            vector = cap_continuation_basis_vector(
                fixed_node, new_neighbor, self.main_uv_index(),
                owner_edges[-1] if owner_edges else None,
                direction_sign=-1.0,
                continuation_direction=start - end,
            )
            return vector
        tangent_position = next(
            handle.pos() for handle in self.bezier_handles
            if handle.role == "end_tangent"
        )
        return np_point(tangent_position - self.bezier_end_position())

    def update_bezier_from_handles(self):
        self.one_cap_global_rows = {}
        self.one_cap_global_active = False
        self._one_cap_baseline_outer = None
        self._one_cap_global_warning_emitted = False
        start_position = self.bezier_start_position()
        end_position = self.bezier_end_position()
        start = np_point(start_position)
        end = np_point(end_position)
        start_vector = self.bezier_start_vector()
        end_vector = self.bezier_end_vector()
        fractions = cumulative_node_length_fractions(self.ordered_nodes)
        points, tangents, parameters = sample_cubic_bezier_by_arc_fractions(
            start, end, start_vector, end_vector, fractions
        )
        self.outer_nodes = [extended_patch_node(qt_point(point)) for point in points]
        fixed_start_node = self.fixed_bezier_start_node()
        fixed_end_node = self.fixed_bezier_end_node()
        if fixed_start_node is not None:
            self.outer_nodes[0] = fixed_start_node
        if fixed_end_node is not None:
            self.outer_nodes[-1] = fixed_end_node
        self.outer_tangents = tangents
        self.outer_parameters = parameters
        if self.one_cap_topology is not None:
            self._one_cap_baseline_outer = {
                "points": [np.array(point, copy=True) for point in points],
                "tangents": [
                    np.array(tangent, copy=True) for tangent in tangents
                ],
                "parameters": np.array(parameters, copy=True),
                "start_vector": np.array(start_vector, copy=True),
                "end_vector": np.array(end_vector, copy=True),
            }
        self.redraw()
        self.sync_global_cap_tangent_handle()

    @staticmethod
    def add_curved_preview_row(path, row, vectors):
        path.moveTo(row[0].position)
        for index in range(len(row) - 1):
            start = row[index].position
            end = row[index + 1].position
            chord = np_point(end - start)
            start_vector = np.asarray(vectors[index], dtype=float)
            end_vector = np.asarray(vectors[index + 1], dtype=float)
            start_sign = np.sign(np.inner(start_vector, chord)) or 1.0
            end_sign = np.sign(np.inner(end_vector, -chord)) or -1.0
            path.cubicTo(
                start + qt_point(start_sign * start_vector),
                end + qt_point(end_sign * end_vector),
                end,
            )

    @staticmethod
    def add_exact_curved_preview_row(path, segments):
        if not segments:
            return
        path.moveTo(qt_point(segments[0][0]))
        for segment in segments:
            path.cubicTo(
                qt_point(segment[1]), qt_point(segment[2]),
                qt_point(segment[3]),
            )

    def redraw(self, rebuild_rows=True):
        path = QPainterPath()
        if (
            self.one_cap_global_curve_only
            and self.one_cap_global_curve_controls is not None
        ):
            controls = np.asarray(self.one_cap_global_curve_controls)
            path.moveTo(qt_point(controls[0]))
            path.cubicTo(
                qt_point(controls[1]), qt_point(controls[2]),
                qt_point(controls[3]),
            )
            control_path = QPainterPath()
            control_path.moveTo(qt_point(controls[0]))
            control_path.lineTo(qt_point(controls[1]))
            control_path.moveTo(qt_point(controls[2]))
            control_path.lineTo(qt_point(controls[3]))
            if self.one_cap_global_control_polygon is not None:
                self.one_cap_global_control_polygon.setPath(control_path)
            if self.outer_nodes:
                outer_row_index = len(self.preview_node_rows) - 1
                for row_index, row in enumerate(self.preview_node_rows):
                    if not row or row_index == outer_row_index:
                        continue
                    if (
                        0 < row_index < outer_row_index
                        and len(self.preview_along_vectors)
                        == len(self.preview_node_rows)
                    ):
                        self.add_curved_preview_row(
                            path, row,
                            self.preview_along_vectors[row_index],
                        )
                    else:
                        path.moveTo(row[0].position)
                        for node in row[1:]:
                            path.lineTo(node.position)
                for row0, row1 in zip(
                    self.preview_node_rows[:-1],
                    self.preview_node_rows[1:],
                ):
                    for node0, node1 in zip(row0, row1):
                        path.moveTo(node0.position)
                        path.lineTo(node1.position)
            self.setPath(path)
            self.update()
            return
        if rebuild_rows:
            self.rebuild_intermediate_preview_rows()
        if self.one_cap_topology is not None and not self.outer_nodes:
            topology = self.one_cap_topology
            if topology.outer_start_node is not None:
                path.moveTo(self.ordered_nodes[0].position)
                path.lineTo(topology.outer_start_node.position)
            else:
                path.moveTo(self.ordered_nodes[-1].position)
                path.lineTo(topology.outer_end_node.position)
        if self.bezier_mode and not self.one_cap_global_active:
            start = self.bezier_start_position()
            end = self.bezier_end_position()
            path.moveTo(start)
            path.cubicTo(
                start + qt_point(self.bezier_start_vector()),
                end + qt_point(self.bezier_end_vector()), end,
            )
        if self.bezier_mode:
            start = self.bezier_start_position()
            end = self.bezier_end_position()
            handles = {handle.role: handle for handle in self.bezier_handles}
            if "start_tangent" in handles:
                path.moveTo(start)
                path.lineTo(handles["start_tangent"].pos())
            if "end_tangent" in handles:
                path.moveTo(end)
                path.lineTo(handles["end_tangent"].pos())
            if "global_start_tangent" in handles:
                path.moveTo(start)
                path.lineTo(handles["global_start_tangent"].pos())
            if "global_end_tangent" in handles:
                path.moveTo(end)
                path.lineTo(handles["global_end_tangent"].pos())
        if self.outer_nodes:
            for row_index, row in enumerate(self.preview_node_rows):
                if (
                    not row
                    or (
                        self.bezier_mode
                        and not self.one_cap_global_active
                        and row_index == len(self.preview_node_rows) - 1
                    )
                ):
                    continue
                if (
                    self.one_cap_global_active
                    and row_index in self.one_cap_global_rows
                ):
                    self.add_exact_curved_preview_row(
                        path, self.one_cap_global_rows[row_index]["segments"]
                    )
                    continue
                if (
                    self.bezier_mode
                    and 0 < row_index < len(self.preview_node_rows) - 1
                    and len(self.preview_along_vectors)
                    == len(self.preview_node_rows)
                ):
                    self.add_curved_preview_row(
                        path, row, self.preview_along_vectors[row_index]
                    )
                else:
                    path.moveTo(row[0].position)
                    for node in row[1:]:
                        path.lineTo(node.position)
            for row0, row1 in zip(
                self.preview_node_rows[:-1], self.preview_node_rows[1:]
            ):
                for node0, node1 in zip(row0, row1):
                    path.moveTo(node0.position)
                    path.lineTo(node1.position)
        self.setPath(path)
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(self.pen())
        painter.drawPath(self.path())


class this_view(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberBand  = None
        self.start_point = None
        self.end_point   = None
        self.rubberband_mode = None
        self.right_pan_active = False
        self.right_pan_last_pos = None
        self.zoom_level  = 1.0
        self.auto_fit_on_resize = True
        self._view_adjustment_in_progress = False
        self.selected_point = None
        self.dragged_node = None
        self.pending_main_uv_index = None
        self.pending_radial_layers = 1
        self.pending_bezier_mode = True
        self.editable_depth = 0
        self._element_adjacency = None
        self.editable_element_indices_set = set()
        self.editable_node_indices_set = set()
        self.patch_controls = None
        self.document_modified = False
        self.document_window = None
        self.wall_outline_points = []
        self.wall_outline_filename = None
        self.wall_outline_item = None
        self.last_geometry_undo = None
        self._geometry_drag_snapshot = None
        self.current_patch = None
        self.current_extended_patch = None
        self.selected_edges = []
        self.selected_nodes = []
        self.selected_elements = []
        self._view_adjustment_in_progress = True
        try:
            self.apply_view_scale(self.zoom_level)
        finally:
            self._view_adjustment_in_progress = False

    def apply_view_scale(self, zoom_level=None):
        """Apply zoom while displaying physical R right and physical Z up."""
        if zoom_level is not None:
            self.zoom_level = float(zoom_level)
        self.setTransform(
            QTransform().scale(self.zoom_level, -self.zoom_level)
        )

    def grid_bounding_rect(self):
        """Return persistent mesh and wall bounds, excluding previews."""
        grid_rect = None
        background = globals().get("static_mesh")
        if graphics_item_in_scene(background, self.scene()):
            grid_rect = background.mapRectToScene(background.boundingRect())
        else:
            for element in globals().get("element_list", []):
                if (
                    not getattr(element, "active", True)
                    or not isinstance(element, QGraphicsItem)
                ):
                    continue
                element_rect = element.mapRectToScene(element.boundingRect())
                grid_rect = (
                    QRectF(element_rect)
                    if grid_rect is None
                    else grid_rect.united(element_rect)
                )

        active_nodes = [
            node for node in globals().get("node_list", [])
            if getattr(node, "active", True)
        ]
        if active_nodes:
            x_values = [node.position.x() for node in active_nodes]
            y_values = [node.position.y() for node in active_nodes]
            node_rect = QRectF(
                min(x_values), min(y_values),
                max(x_values) - min(x_values),
                max(y_values) - min(y_values),
            )
            grid_rect = (
                node_rect
                if grid_rect is None
                else grid_rect.united(node_rect)
            )
        wall_item = self.wall_outline_item
        if graphics_item_in_scene(wall_item, self.scene()):
            wall_rect = wall_item.mapRectToScene(wall_item.boundingRect())
            grid_rect = (
                QRectF(wall_rect)
                if grid_rect is None
                else grid_rect.united(wall_rect)
            )
        return grid_rect

    def rebuild_wall_outline_item(self):
        """Recreate the stored wall reference in the current scene."""
        self.wall_outline_item = None
        if self.scene() is None or not self.wall_outline_points:
            return None
        item = wall_outline_item(self.wall_outline_points, this_scaling)
        self.scene().addItem(item)
        self.wall_outline_item = item
        return item

    def fit_grid_to_window(self):
        self.auto_fit_on_resize = True
        grid_rect = self.grid_bounding_rect()
        if grid_rect is None or grid_rect.isEmpty():
            return

        width = grid_rect.width()
        height = grid_rect.height()
        scale = max(width, height, 1.0)
        margin_x = 0.05 * (width if width > 0.0 else scale)
        margin_y = 0.05 * (height if height > 0.0 else scale)
        fit_rect = grid_rect.adjusted(
            -margin_x, -margin_y, margin_x, margin_y
        )
        # When the whole scene fits, QGraphicsView has no scrollbar range and
        # aligns sceneRect() rather than the rectangle passed to fitInView().
        # Graphics handles can make those rectangles have different centers.
        # Use the persistent fit bounds as the fitted-mode scrollable rectangle
        # as well, avoiding both that mismatch and scrollbar visibility churn.
        fit_center = fit_rect.center()
        self._view_adjustment_in_progress = True
        try:
            self.setSceneRect(fit_rect)
            self.apply_view_scale(1.0)
            self.fitInView(fit_rect, Qt.KeepAspectRatio)
            self.centerOn(fit_center)
            transform = self.transform()
            self.zoom_level = math.hypot(transform.m11(), transform.m12())
        finally:
            self._view_adjustment_in_progress = False

    def restore_full_scene_rect(self):
        """Restore all-item scroll bounds before leaving fitted view mode."""
        if self.scene() is None:
            return
        center = self.mapToScene(self.viewport().rect().center())
        self._view_adjustment_in_progress = True
        try:
            self.setSceneRect(self.scene().sceneRect())
            self.centerOn(center)
        finally:
            self._view_adjustment_in_progress = False

    def reset_zoom(self):
        print("resetting zoom_level to 1")
        self.auto_fit_on_resize = False
        self.restore_full_scene_rect()
        self._view_adjustment_in_progress = True
        try:
            self.apply_view_scale(1.0)
        finally:
            self._view_adjustment_in_progress = False

    def resizeEvent(self, event):
        keep_fitted = getattr(self, "auto_fit_on_resize", True)
        self._view_adjustment_in_progress = True
        try:
            super().resizeEvent(event)
        finally:
            self._view_adjustment_in_progress = False
        if keep_fitted:
            self.fit_grid_to_window()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if not getattr(self, "_view_adjustment_in_progress", False):
            self.auto_fit_on_resize = False

    def clear_selection(self):
        self.finish_rubber_band()
        self.replace_boundary_edge_selection([])
        if (
            self.current_patch is not None
            and isinstance(self.current_patch, QGraphicsItem)
        ):
            self.scene().removeItem(self.current_patch)
        self.current_patch = None
        if (
            self.current_extended_patch is not None
            and isinstance(self.current_extended_patch, QGraphicsItem)
        ):
            self.scene().removeItem(self.current_extended_patch)
        self.current_extended_patch = None
        self.pending_main_uv_index = None
        self.update_patch_controls()

    def set_edge_selected(self, edge, selected):
        edge.setPen(
            boundary_edge_pen(Qt.green, 2.5)
            if selected else boundary_edge_pen()
        )
        edge.update()

    def refresh_selection_highlights(self):
        for element in self.selected_elements:
            element.path_item.setBrush(QBrush(QColor(255, 255, 255, 64)))
            element.update()
        for node in self.selected_nodes:
            node.ellipse_item.setBrush(QBrush(NODE_COLOR))
            node.update()

        self.selected_nodes = []
        self.selected_elements = []
        for edge in self.selected_edges:
            for node in edge.nodes:
                if (
                    isinstance(node, jorek_node_item)
                    and getattr(node, "active", True)
                    and node not in self.selected_nodes
                ):
                    self.selected_nodes.append(node)
            element = element_by_index(edge.element_index)
            if (
                isinstance(element, jorek_element_item)
                and getattr(element, "active", True)
                and element not in self.selected_elements
            ):
                self.selected_elements.append(element)

        for node in self.selected_nodes:
            node.ellipse_item.setBrush(QBrush(SELECTED_NODE_COLOR))
            node.update()
        for element in self.selected_elements:
            element.path_item.setBrush(QBrush(QColor(50, 50, 50, 64)))
            element.update()

    def replace_boundary_edge_selection(self, edges):
        for edge in self.selected_edges:
            self.set_edge_selected(edge, False)
        self.selected_edges = []
        for edge in edges:
            if (
                getattr(edge, "active", True)
                and edge not in self.selected_edges
            ):
                self.selected_edges.append(edge)
                self.set_edge_selected(edge, True)
        self.refresh_selection_highlights()

    def add_boundary_edge_selection(self, edges):
        for edge in edges:
            if (
                getattr(edge, "active", True)
                and edge not in self.selected_edges
            ):
                self.selected_edges.append(edge)
                self.set_edge_selected(edge, True)
        self.refresh_selection_highlights()

    def toggle_boundary_edge_selection(self, edges):
        processed = []
        for edge in edges:
            if not getattr(edge, "active", True) or edge in processed:
                continue
            processed.append(edge)
            if edge in self.selected_edges:
                self.selected_edges.remove(edge)
                self.set_edge_selected(edge, False)
            else:
                self.selected_edges.append(edge)
                self.set_edge_selected(edge, True)
        self.refresh_selection_highlights()

    def finish_rubber_band(self):
        if self.rubberBand is not None:
            self.rubberBand.hide()
            self.rubberBand.deleteLater()
        self.rubberBand = None
        self.start_point = None
        self.end_point = None
        self.rubberband_mode = None

    def set_patch_status(self, message):
        if self.patch_controls is not None:
            self.patch_controls.set_status(message)

    def mark_document_modified(self):
        self.document_modified = True
        if self.document_window is not None:
            self.document_window.update_window_title()

    def clear_geometry_undo(self):
        self.last_geometry_undo = None
        self._geometry_drag_snapshot = None

    def begin_geometry_drag(self, node, edit_kind):
        self._geometry_drag_snapshot = None
        node_index = getattr(node, "index", None)
        nodes_xx = getattr(globals().get("jorek"), "nodes_xx", None)
        if (
            node_index is None
            or nodes_xx is None
            or node_index < 0
            or node_index >= nodes_xx.shape[2]
        ):
            return
        self._geometry_drag_snapshot = {
            "node_index": int(node_index),
            "nodes_xx": np.array(nodes_xx[:, :, node_index], copy=True),
            "kind": edit_kind,
        }

    def complete_geometry_drag(self):
        snapshot = self._geometry_drag_snapshot
        self._geometry_drag_snapshot = None
        if snapshot is None:
            return False
        node_index = snapshot["node_index"]
        nodes_xx = getattr(globals().get("jorek"), "nodes_xx", None)
        if (
            nodes_xx is None
            or node_index < 0
            or node_index >= nodes_xx.shape[2]
        ):
            return False
        if np.array_equal(
            nodes_xx[:, :, node_index], snapshot["nodes_xx"]
        ):
            return False
        self.last_geometry_undo = snapshot
        return True

    def refresh_geometry_for_node(self, node_index):
        nodes_xx = getattr(globals().get("jorek"), "nodes_xx", None)
        if (
            nodes_xx is None
            or node_index < 0
            or node_index >= nodes_xx.shape[2]
            or node_index >= len(globals().get("node_list", []))
        ):
            return False
        node = node_list[node_index]
        if getattr(node, "index", None) != node_index:
            return False
        if isinstance(node, QGraphicsItem):
            node.prepareGeometryChange()
        node.xx = this_scaling * np.array(
            nodes_xx[:, :, node_index], copy=True
        )
        node.position = qt_point(node.xx[:, 0])
        if isinstance(node, jorek_node_item):
            node.ellipse_item.setPos(node.position)
            node.blue_handle.sync_position()
            node.red_handle.sync_position()
            node.update_connected_items()
        else:
            node.update()
        if self.current_extended_patch is not None:
            self.current_extended_patch.redraw()
        if self.current_patch is not None:
            self.current_patch.update()
        rebuild_static_mesh_path(self.scene())
        if self.scene() is not None:
            self.scene().update()
        return True

    def undo_last_geometry_edit(self):
        if self.dragged_node is not None or self.selected_point is not None:
            return False
        snapshot = self.last_geometry_undo
        if snapshot is None:
            return False
        self.last_geometry_undo = None
        node_index = snapshot["node_index"]
        nodes_xx = getattr(globals().get("jorek"), "nodes_xx", None)
        if (
            nodes_xx is None
            or node_index < 0
            or node_index >= nodes_xx.shape[2]
            or nodes_xx[:, :, node_index].shape != snapshot["nodes_xx"].shape
        ):
            return False
        nodes_xx[:, :, node_index] = snapshot["nodes_xx"]
        if not self.refresh_geometry_for_node(node_index):
            return False
        self.mark_document_modified()
        self.set_patch_status(
            "Undid node move"
            if snapshot["kind"] == "node" else "Undid basis-vector move"
        )
        return True

    def update_patch_controls(self):
        if self.patch_controls is not None:
            self.patch_controls.update_from_view()

    def update_manual_outer_node_status(self):
        patch = self.current_extended_patch
        if patch is None or patch.bezier_mode:
            return
        defined_count = len(patch.outer_nodes)
        required_count = patch.required_outer_node_count
        if defined_count == required_count:
            self.set_patch_status("Straight patch ready")
        elif patch.one_cap_topology is not None:
            self.set_patch_status("Ctrl-click free outer endpoint")
        else:
            self.set_patch_status(
                "Add outer nodes with Ctrl-click ({}/{})".format(
                    defined_count, required_count
                )
            )

    def set_pending_main_uv_index(self, value):
        if value not in (None, 1, 2):
            raise ValueError("Extended-patch main direction must be Auto, 1, or 2")
        self.pending_main_uv_index = value
        if value is not None:
            print("extended patch pending main uv_index:", value)
        self.update_patch_controls()

    def set_extended_radial_layers(self, nr):
        if not 1 <= nr <= MAX_RADIAL_LAYERS:
            raise ValueError(
                "Radial layer count must be between 1 and {}".format(
                    MAX_RADIAL_LAYERS
                )
            )
        patch = self.current_extended_patch
        if patch is None:
            self.pending_radial_layers = nr
        else:
            topology = patch.one_cap_topology or patch.capped_gap
            has_caps = topology is not None and bool(
                getattr(topology, "start_cap_edges", [])
                or getattr(topology, "end_cap_edges", [])
            )
            patch.set_radial_layers(nr)
            if not has_caps:
                self.pending_radial_layers = nr
            print("extended patch radial layers:", patch.radial_layers)
        self.update_patch_controls()

    def set_pending_bezier_mode(self, enabled):
        self.pending_bezier_mode = bool(enabled)
        self.update_patch_controls()

    def invalidate_element_adjacency(self):
        self._element_adjacency = None

    def set_editable_depth(self, depth):
        depth = int(depth)
        if not 0 <= depth <= 9:
            raise ValueError("Editable depth must be between 0 and 9")
        if depth == self.editable_depth:
            self.update_patch_controls()
            return

        saved_transform = QTransform(self.transform())
        saved_center = self.mapToScene(self.viewport().rect().center())
        self.clear_selection()
        self.selected_point = None
        self.dragged_node = None
        self.editable_depth = depth
        if self.scene() is not None and self.scene() is globals().get("scene"):
            rebuild_graphics_layers(active_view=self)
            saved_auto_fit = self.auto_fit_on_resize
            self._view_adjustment_in_progress = True
            try:
                self.setTransform(saved_transform)
                self.centerOn(saved_center)
            finally:
                self._view_adjustment_in_progress = False
                self.auto_fit_on_resize = saved_auto_fit
        self.set_patch_status("Editable depth {}".format(depth))
        self.update_patch_controls()

    def enable_current_patch_bezier(self):
        if self.current_extended_patch is None:
            print("Start an extended patch before enabling Bézier mode")
            self.set_patch_status("Start an extended patch first")
            self.update_patch_controls()
            return False
        if self.current_extended_patch.bezier_mode:
            self.update_patch_controls()
            return False
        self.current_extended_patch.enable_bezier_mode(
            one_cap_curve_only=True
        )
        self.pending_bezier_mode = True
        self.set_patch_status("Bézier enabled")
        self.update_patch_controls()
        return True

    def commit_current_patch(self):
        print("convert patch to nodes and elements")
        if self.current_extended_patch is not None:
            add_extended_patch_to_nodes_elements(self.current_extended_patch)
            committed = self.current_extended_patch is None
            if committed:
                rebuild_graphics_layers(
                    active_view=self, topology_changed=True
                )
                self.pending_main_uv_index = None
                self.pending_bezier_mode = True
                self.set_patch_status("Patch committed")
                self.mark_document_modified()
            self.update_patch_controls()
            return committed
        if self.current_patch is None:
            print("No valid patch has been defined")
            self.set_patch_status("No patch to commit")
            self.update_patch_controls()
            return False
        add_patch_to_nodes_elements(self.current_patch)
        rebuild_graphics_layers(active_view=self, topology_changed=True)
        self.pending_bezier_mode = True
        self.mark_document_modified()
        self.update_patch_controls()
        return True

    def cancel_current_operation(self):
        self.clear_selection()
        self.pending_bezier_mode = True
        self.set_patch_status("Ready")
        self.update_patch_controls()

    def create_extended_patch_preview(self):
        if self.current_extended_patch is not None:
            self.set_patch_status("Patch preview already exists")
            self.update_patch_controls()
            return False
        boundary_topology = None
        ambiguous_topology = False
        for edge in self.selected_edges:
            print(
                "selected edge:", edge.vertices,
                "uv =", edge.uv_index,
                "element =", edge.element_index,
                "side =", getattr(edge, "element_side", None),
            )
        main_uv_index = self.pending_main_uv_index
        if main_uv_index is not None:
            print("extended patch main uv_index:", main_uv_index)
        try:
            boundary_topology = ordered_extended_boundary_topology(
                self.selected_edges, main_uv_index=main_uv_index
            )
        except ValueError as error:
            print("extended topology detection failed:", error)
            if (
                main_uv_index is None
                and "Ambiguous extended patch" in str(error)
            ):
                ambiguous_topology = True
                print(
                    "Press 1 or 2 to choose the extended-patch main "
                    "boundary direction, then press E again"
                )
                self.set_patch_status(
                    "Ambiguous boundary: choose Direction 1 or 2"
                )
            else:
                self.set_patch_status(str(error))
            boundary_topology = None
        if boundary_topology is not None:
            ordered_nodes = boundary_topology.inner_nodes
            ordered_edges = boundary_topology.inner_edges
        else:
            if len({edge.uv_index for edge in self.selected_edges}) > 1:
                print(
                    "Extended topology with perpendicular side edges "
                    "was not recognized; patch creation aborted"
                )
                self.update_patch_controls()
                return False
            selection_error = validate_boundary_chain(self.selected_edges)
            if selection_error:
                print(selection_error)
                if not self.selected_edges:
                    self.set_patch_status("Select boundary edges")
                elif not ambiguous_topology:
                    self.set_patch_status(selection_error)
                self.update_patch_controls()
                return False
            ordered_nodes, ordered_edges = ordered_edge_chain(
                self.selected_edges
            )
        if (
            boundary_topology is not None
            and boundary_topology.start_cap_edges
            and boundary_topology.end_cap_edges
        ):
            topology_error = two_cap_topology_error(
                boundary_topology, ordered_nodes
            )
            if topology_error:
                print(topology_error)
                self.set_patch_status(topology_error)
                self.update_patch_controls()
                return False
        if (
            self.current_patch is not None
            and isinstance(self.current_patch, QGraphicsItem)
        ):
            self.scene().removeItem(self.current_patch)
        self.current_patch = None
        if (
            self.current_extended_patch is not None
            and isinstance(self.current_extended_patch, QGraphicsItem)
        ):
            self.scene().removeItem(self.current_extended_patch)
        self.current_extended_patch = extended_patch(
            ordered_nodes, ordered_edges, can_commit=True,
        )
        if (
            boundary_topology is not None
            and boundary_topology.start_cap_edge is not None
            and boundary_topology.end_cap_edge is not None
        ):
            self.current_extended_patch.capped_gap = boundary_topology
            self.current_extended_patch.radial_layers = len(
                boundary_topology.start_cap_edges
            )
            segment_count = len(ordered_edges)
            start = boundary_topology.outer_start_node.position
            end = boundary_topology.outer_end_node.position
            self.current_extended_patch.set_outer_positions(
                interpolate_positions(start, end, segment_count)
            )
        elif (
            boundary_topology is not None
            and (
                boundary_topology.start_cap_edge is not None
                or boundary_topology.end_cap_edge is not None
            )
        ):
            self.current_extended_patch.one_cap_topology = boundary_topology
            cap_edges = (
                boundary_topology.start_cap_edges
                or boundary_topology.end_cap_edges
            )
            self.current_extended_patch.radial_layers = len(cap_edges)
            self.current_extended_patch.redraw()
        else:
            self.current_extended_patch.set_radial_layers(
                self.pending_radial_layers
            )
        self.scene().addItem(self.current_extended_patch)
        self.pending_main_uv_index = None
        print(
            "extended patch ordered node indices:",
            [node.index for node in ordered_nodes],
        )
        print(
            "extended patch ordered edge vertex pairs:",
            [list(edge.vertices) for edge in ordered_edges],
        )
        if self.pending_bezier_mode:
            self.enable_current_patch_bezier()
            self.set_patch_status("Bézier preview created")
        else:
            self.update_manual_outer_node_status()
            self.update_patch_controls()
        return True


    def keyPressEvent(self, event):
        print("key pressed: ",event.key())
        modifiers = (
            event.modifiers()
            if hasattr(event, "modifiers")
            else QApplication.keyboardModifiers()
        )
        if (
            event.key() == Qt.Key_Z
            and modifiers & Qt.ControlModifier
        ):
            self.undo_last_geometry_edit()
            return
        if (
            self.current_extended_patch is not None
            and Qt.Key_1 <= event.key() <= Qt.Key_9
        ):
            nr = event.key() - Qt.Key_0
            self.set_extended_radial_layers(nr)
            return
        if (
            self.current_extended_patch is None
            and event.key() in (Qt.Key_1, Qt.Key_2)
        ):
            self.set_pending_main_uv_index(event.key() - Qt.Key_0)
            return
        # A containing grid_editor_window owns these shortcuts.  Keep this
        # fallback for standalone views and direct programmatic key events,
        # while ensuring a real window key press has only one command owner.
        if self.document_window is None:
            if event.key() == Qt.Key_F:
                print("fit grid to window")
                self.fit_grid_to_window()
                return
            if event.key() == Qt.Key_Escape:
                self.cancel_current_operation()
                return
            if event.key() == Qt.Key_U:
                self.reset_zoom()
                return
            if event.key() == Qt.Key_P:
                self.commit_current_patch()
                return
            if event.key() == Qt.Key_B:
                self.enable_current_patch_bezier()
                return
            if event.key() == Qt.Key_E:
                self.create_extended_patch_preview()
                return

        if event.key() == Qt.Key_Delete:
            delete_selected_element(self)
            self.update_patch_controls()
            return

    def mousePressEvent(self, event):        

        if hasattr(event, "button") and event.button() == Qt.RightButton:
            self.right_pan_active = True
            self.right_pan_last_pos = event.pos()
            self.auto_fit_on_resize = False
            self.restore_full_scene_rect()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        modifiers = (
            event.modifiers()
            if hasattr(event, "modifiers")
            else QApplication.keyboardModifiers()
        )
        if modifiers == Qt.ControlModifier:
            if self.current_extended_patch is not None:
                if not self.current_extended_patch.add_outer_node(
                    self.mapToScene(event.pos())
                ):
                    print("All extended-patch outer points have been defined")
                self.update_manual_outer_node_status()
                self.update_patch_controls()
                return
            if len(self.selected_edges) == 3:
                if self.current_patch is None:
                    selection_error = validate_single_element_patch(
                        self.selected_edges
                    )
                    if selection_error:
                        print(selection_error)
                        return
                    self.current_patch = big_patch()
                    self.scene().addItem(self.current_patch)
                    self.current_patch.update()
                return
            if self.current_patch is None:
                selection_error = validate_single_element_patch(
                    view.selected_edges
                )
                if selection_error:
                    print(selection_error)
                    return
                for selected_edge in view.selected_edges:
                    print(
                        "starting patch from edge: element_index =",
                        selected_edge.element_index,
                        "element_side =",
                        selected_edge.element_side,
                        "vertices =",
                        selected_edge.vertices,
                    )
                print("define new patch nodes")
                self.current_patch = big_patch()
                self.scene().addItem(self.current_patch)

            this_node = big_patch_node(view.mapToScene(event.pos()))
            self.current_patch.add_corner(this_node)
            self.current_patch.update()
        else:    
            items = self.items(event.pos())
            for item in items:
                if (
                    isinstance(item, extended_bezier_handle)
                    or (
                        isinstance(item, basis_vector_handle)
                        and getattr(item.node, "active", True)
                    )
                ):
                    self.selected_point = item
                    if isinstance(item, basis_vector_handle):
                        self.begin_geometry_drag(
                            item.node, "basis-vector"
                        )
                    event.accept()
                    return
                if (
                    isinstance(item, jorek_node_item)
                    and getattr(item, "active", True)
                ):
                    self.dragged_node = item
                    self.begin_geometry_drag(item, "node")
                    event.accept()
                    return
      
            self.start_point = event.pos()
            if (
                modifiers & Qt.ShiftModifier
                and modifiers & Qt.ControlModifier
            ):
                self.rubberband_mode = "toggle"
            elif modifiers & Qt.ShiftModifier:
                self.rubberband_mode = "add"
            else:
                self.rubberband_mode = None
            self.rubberBand = QRubberBand(
                QRubberBand.Rectangle, self.viewport()
            )
            self.rubberBand.setGeometry(QRect(self.start_point, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):
        if self.right_pan_active:
            position = event.pos()
            delta = position - self.right_pan_last_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.right_pan_last_pos = position
            event.accept()
            return
        if self.selected_point:
            self.selected_point.move_to_scene(self.mapToScene(event.pos()))
            event.accept()
            return
        if self.dragged_node is not None:
            self.dragged_node.move_to_scene(self.mapToScene(event.pos()))
            event.accept()
            return
        if self.rubberBand and self.start_point: 
           self.rubberBand.setGeometry(
               QRect(self.start_point, event.pos()).normalized()
           )
           self.end_point = event.pos()
           return

    def mouseReleaseEvent(self, event):
        if (
            self.right_pan_active
            and hasattr(event, "button")
            and event.button() == Qt.RightButton
        ):
            self.right_pan_active = False
            self.right_pan_last_pos = None
            self.viewport().unsetCursor()
            event.accept()
            return
        if self.selected_point:
            if isinstance(self.selected_point, basis_vector_handle):
                self.complete_geometry_drag()
            self.selected_point = None
            rebuild_static_mesh_path(self.scene())
            event.accept()
            return
        if self.dragged_node is not None:
            self.complete_geometry_drag()
            self.dragged_node = None
            rebuild_static_mesh_path(self.scene())
            event.accept()
            return
        if (not self.rubberBand) or (not self.end_point):
            self.finish_rubber_band()
            return

        zoom_rect = QRect(self.start_point, self.end_point).normalized()

        if self.rubberband_mode in ("add", "toggle"):
            items = self.items(zoom_rect)
            hit_edges = [
                item for item in items
                if (
                    isinstance(item, boundary_edge)
                    and getattr(item, "active", True)
                )
            ]
            if self.rubberband_mode == "add":
                print("shift key pressed: add to edge selection")
                self.add_boundary_edge_selection(hit_edges)
            else:
                print("ctrl+shift pressed: toggle edge selection")
                self.toggle_boundary_edge_selection(hit_edges)
            self.update_patch_controls()

        else:
            if not zoom_rect.isEmpty():
                self.auto_fit_on_resize = False
                zoom_center = self.mapToScene(zoom_rect.center())
                self.restore_full_scene_rect()
                viewport_rect = self.viewport().rect()
                zoom_factor_x = viewport_rect.width()  / zoom_rect.width()
                zoom_factor_y = viewport_rect.height() / zoom_rect.height()
                zoom_factor   = min(zoom_factor_x, zoom_factor_y)
                self.zoom_level *= zoom_factor
 
                self.centerOn(zoom_center)
                self.apply_view_scale(self.zoom_level)
        self.finish_rubber_band()


class extended_patch_controls(QGroupBox):
    """Compact controls for the existing extended-patch view state machine."""
    def __init__(self, view=None, parent=None):
        super().__init__("Extended patch", parent)
        self.view = None
        self._updating = False
        self.status_message = "Ready"
        self.setFixedWidth(250)

        layout = QVBoxLayout(self)
        self.selection_label = QLabel("Selection:\n0 boundary edges")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        layout.addWidget(QLabel("Editable depth:"))
        self.editable_depth_spin = QSpinBox()
        self.editable_depth_spin.setRange(0, 9)
        self.editable_depth_spin.setToolTip(
            "Number of element-neighbour layers inward from the boundary "
            "that are interactively editable."
        )
        layout.addWidget(self.editable_depth_spin)
        self.editable_region_label = QLabel("Editable region:\n0 elements, 0 nodes")
        self.editable_region_label.setWordWrap(True)
        layout.addWidget(self.editable_region_label)

        layout.addWidget(QLabel("Main direction:"))
        self.main_direction_combo = QComboBox()
        self.main_direction_combo.addItem("Auto", None)
        self.main_direction_combo.addItem("Direction 1 (u)", 1)
        self.main_direction_combo.addItem("Direction 2 (v)", 2)
        layout.addWidget(self.main_direction_combo)

        layout.addWidget(QLabel("Radial layers:"))
        self.radial_layers_spin = QSpinBox()
        self.radial_layers_spin.setRange(1, MAX_RADIAL_LAYERS)
        layout.addWidget(self.radial_layers_spin)
        self.radial_note_label = QLabel("")
        layout.addWidget(self.radial_note_label)

        layout.addWidget(QLabel("Outer boundary:"))
        self.outer_boundary_combo = QComboBox()
        self.outer_boundary_combo.addItem("Straight", False)
        self.outer_boundary_combo.addItem("Bézier", True)
        layout.addWidget(self.outer_boundary_combo)

        self.preview_button = QPushButton("Create / Preview")
        self.commit_button = QPushButton("Commit")
        self.cancel_button = QPushButton("Cancel")
        self.fit_button = QPushButton("Fit to window")
        self.load_wall_button = QPushButton("Load wall outline")
        self.wall_label = QLabel("Wall: none")
        self.wall_label.setWordWrap(True)
        layout.addWidget(self.fit_button)
        layout.addWidget(self.load_wall_button)
        layout.addWidget(self.wall_label)
        layout.addWidget(self.preview_button)
        button_row = QHBoxLayout()
        button_row.addWidget(self.commit_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.main_direction_combo.currentIndexChanged.connect(
            self._main_direction_changed
        )
        self.editable_depth_spin.valueChanged.connect(
            self._editable_depth_changed
        )
        self.radial_layers_spin.valueChanged.connect(
            self._radial_layers_changed
        )
        self.outer_boundary_combo.currentIndexChanged.connect(
            self._outer_boundary_changed
        )
        self.preview_button.clicked.connect(self._create_preview)
        self.commit_button.clicked.connect(self._commit)
        self.cancel_button.clicked.connect(self._cancel)
        self.fit_button.clicked.connect(self._fit_to_window)
        self.load_wall_button.clicked.connect(self._load_wall_outline)
        if view is not None:
            self.attach_view(view)
        else:
            self.update_from_view()

    def attach_view(self, view):
        if self.view is not None and self.view.patch_controls is self:
            self.view.patch_controls = None
        self.view = view
        view.patch_controls = self
        self.update_from_view()

    def set_status(self, message):
        self.status_message = message
        self.status_label.setText("Status: " + message)

    def _active_topology(self):
        if self.view is None or self.view.current_extended_patch is None:
            return None
        patch = self.view.current_extended_patch
        return patch.one_cap_topology or patch.capped_gap

    def _has_cap_chain(self):
        topology = self._active_topology()
        return topology is not None and bool(
            getattr(topology, "start_cap_edges", [])
            or getattr(topology, "end_cap_edges", [])
        )

    def _selection_summary(self):
        if self.view is None:
            return "Selection:\n0 boundary edges"
        patch = self.view.current_extended_patch
        if patch is None:
            return "Selection:\n{} boundary edges".format(
                len(self.view.selected_edges or [])
            )
        lines = ["Selection:", "{} main edges".format(len(patch.ordered_edges))]
        topology = patch.one_cap_topology or patch.capped_gap
        if topology is None:
            lines.append("no caps")
        else:
            start_count = len(getattr(topology, "start_cap_edges", []))
            end_count = len(getattr(topology, "end_cap_edges", []))
            if start_count and end_count:
                lines.append("two caps: {} + {} edges".format(
                    start_count, end_count
                ))
            else:
                lines.append("one cap: {} edges".format(
                    start_count or end_count
                ))
        lines.append("{} radial layers".format(patch.radial_layers))
        lines.append("main direction: {}".format(patch.main_uv_index()))
        return "\n".join(lines)

    def update_from_view(self):
        self._updating = True
        try:
            if self.view is None:
                self.setEnabled(False)
                return
            self.setEnabled(True)
            patch = self.view.current_extended_patch
            self.selection_label.setText(self._selection_summary())
            self.editable_depth_spin.setValue(self.view.editable_depth)
            self.editable_region_label.setText(
                "Editable region:\ndepth {}, {} elements, {} nodes".format(
                    self.view.editable_depth,
                    len(self.view.editable_element_indices_set),
                    len(self.view.editable_node_indices_set),
                )
            )

            direction = (
                patch.main_uv_index()
                if patch is not None else self.view.pending_main_uv_index
            )
            direction_index = {None: 0, 1: 1, 2: 2}[direction]
            self.main_direction_combo.setCurrentIndex(direction_index)
            self.main_direction_combo.setEnabled(patch is None)

            radial_layers = (
                patch.radial_layers
                if patch is not None else self.view.pending_radial_layers
            )
            self.radial_layers_spin.setValue(radial_layers)
            cap_fixed = patch is not None and self._has_cap_chain()
            self.radial_layers_spin.setEnabled(not cap_fixed)
            self.radial_note_label.setText(
                "(fixed by cap chain)" if cap_fixed else ""
            )

            bezier_active = patch is not None and patch.bezier_mode
            desired_bezier = (
                bezier_active
                or (patch is None and self.view.pending_bezier_mode)
            )
            self.outer_boundary_combo.setCurrentIndex(
                1 if desired_bezier else 0
            )
            self.outer_boundary_combo.setEnabled(patch is None)
            self.preview_button.setEnabled(patch is None)
            self.commit_button.setEnabled(
                patch is not None
                and getattr(patch, "can_commit", True)
                and (
                    not hasattr(patch, "outer_nodes")
                    or len(patch.outer_nodes)
                    == patch.required_outer_node_count
                )
            )
        finally:
            self._updating = False

    def _main_direction_changed(self, unused_index):
        if not self._updating and self.view is not None:
            self.view.set_pending_main_uv_index(
                self.main_direction_combo.currentData()
            )

    def _editable_depth_changed(self, value):
        if not self._updating and self.view is not None:
            self.view.set_editable_depth(value)

    def _radial_layers_changed(self, value):
        if not self._updating and self.view is not None:
            self.view.set_extended_radial_layers(value)

    def _outer_boundary_changed(self, unused_index):
        if not self._updating and self.view is not None:
            self.view.set_pending_bezier_mode(bool(
                self.outer_boundary_combo.currentData()
            ))

    def _create_preview(self):
        if self.view is not None:
            self.view.create_extended_patch_preview()

    def _commit(self):
        if self.view is not None:
            self.view.commit_current_patch()

    def _cancel(self):
        if self.view is not None:
            self.view.cancel_current_operation()

    def _fit_to_window(self):
        if self.view is not None:
            self.view.fit_grid_to_window()

    def _load_wall_outline(self):
        if self.view is not None and self.view.document_window is not None:
            self.view.document_window.load_wall_outline_dialog()

    def update_wall_label(self):
        if self.view is None or not self.view.wall_outline_points:
            self.wall_label.setText("Wall: none")
            return
        self.wall_label.setText(
            "Wall: {}\n{} points".format(
                os.path.basename(self.view.wall_outline_filename),
                len(self.view.wall_outline_points),
            )
        )


class grid_editor_window(QMainWindow):
    def __init__(self, graphics_view=None, parent=None):
        super().__init__(parent)
        self.view = graphics_view or this_view()
        self.view.document_window = self
        self.current_filename = None
        self.source_is_grid_only = True
        self.patch_controls = extended_patch_controls(self.view)
        central_widget = QWidget(self)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.patch_controls, 0)
        layout.addWidget(self.view, 1)
        self.setCentralWidget(central_widget)
        self._create_file_menu()
        self._create_command_shortcuts()
        QApplication.instance().installEventFilter(self)
        self.update_window_title()

    def _create_file_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        self.open_action = QAction("&Open...", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_grid_dialog)
        file_menu.addAction(self.open_action)
        self.save_action = QAction("&Save", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.save_grid_from_menu)
        file_menu.addAction(self.save_action)
        self.save_as_action = QAction("Save &As...", self)
        self.save_as_action.setShortcut("Ctrl+Shift+S")
        self.save_as_action.triggered.connect(self.save_grid_as_dialog)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        self.exit_action = QAction("E&xit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

    def _create_command_shortcuts(self):
        command_specs = (
            ("fit", "Fit grid to window", Qt.Key_F,
             self.view.fit_grid_to_window),
            ("preview", "Create extended patch preview", Qt.Key_E,
             self.view.create_extended_patch_preview),
            ("bezier", "Enable patch Bezier mode", Qt.Key_B,
             self.view.enable_current_patch_bezier),
            ("commit", "Commit current patch", Qt.Key_P,
             self.view.commit_current_patch),
            ("reset_zoom", "Reset zoom", Qt.Key_U,
             self.view.reset_zoom),
            ("cancel", "Cancel current operation", Qt.Key_Escape,
             self.view.cancel_current_operation),
        )
        self.command_actions = {}
        self._command_actions_by_key = {}
        for name, text, key, callback in command_specs:
            action = QAction(text, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WindowShortcut)
            action.triggered.connect(
                lambda unused_checked=False, command=callback: command()
            )
            self.addAction(action)
            self.command_actions[name] = action
            self._command_actions_by_key[key] = action

    def eventFilter(self, watched, event):
        # QSpinBox consumes ordinary letter key presses before a
        # Qt.WindowShortcut can fire.  Route only this window's unmodified
        # command keys as a fallback; QAction remains the normal command path.
        if (
            event.type() == QEvent.KeyPress
            and event.modifiers() == Qt.NoModifier
            and hasattr(watched, "window")
            and watched.window() is self
        ):
            action = self._command_actions_by_key.get(event.key())
            if action is not None:
                if isinstance(watched, (QLineEdit, QPlainTextEdit, QTextEdit)):
                    parent = watched.parentWidget()
                    if not isinstance(parent, QAbstractSpinBox):
                        return super().eventFilter(watched, event)
                action.trigger()
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def update_window_title(self):
        title = "JOREK Grid Editor"
        if self.current_filename:
            title += " — " + os.path.basename(self.current_filename)
        if self.view.document_modified:
            title += " *"
        self.setWindowTitle(title)

    def _report_file_error(self, title, error, interactive):
        message = str(error)
        print(title + ":", message)
        self.view.set_patch_status(message)
        if interactive:
            QMessageBox.critical(self, title, message)

    def install_wall_outline(self, points, filename):
        """Atomically replace the wall overlay without modifying grid state."""
        if self.current_filename is None or self.view.scene() is None:
            raise ValueError("Open a grid before loading the wall outline")
        new_item = wall_outline_item(points, this_scaling)
        current_scene = self.view.scene()
        current_scene.addItem(new_item)
        old_item = self.view.wall_outline_item
        if graphics_item_in_scene(old_item, current_scene):
            current_scene.removeItem(old_item)
        self.view.wall_outline_points = list(points)
        self.view.wall_outline_filename = os.path.abspath(filename)
        self.view.wall_outline_item = new_item
        self.patch_controls.update_wall_label()
        message = "Wall loaded: {} points".format(len(points))
        self.view.set_patch_status(message)
        self.view.fit_grid_to_window()
        return True

    def load_wall_outline_file(self, filename, interactive=False):
        try:
            if self.current_filename is None or self.view.scene() is None:
                raise ValueError("Open a grid before loading the wall outline")
            points = parse_wall_outline(filename)
            return self.install_wall_outline(points, filename)
        except (OSError, ValueError) as error:
            self._report_file_error(
                "Could not load wall outline", error, interactive
            )
            return False

    def load_wall_outline_dialog(self):
        filename, unused_filter = QFileDialog.getOpenFileName(
            self, "Load wall outline", "",
            "Text files (*.txt *.dat);;All files (*)",
        )
        return bool(filename) and self.load_wall_outline_file(
            filename, interactive=True
        )

    def open_grid_file(self, filename, interactive=False):
        global jorek, scene, view, node_list, element_list, boundary_list
        global _diagnostic_nodes_xx_before
        old_state = (
            globals().get("jorek"), globals().get("scene"), globals().get("view"),
            globals().get("node_list"), globals().get("element_list"),
            globals().get("boundary_list"), self.view.scene(),
        )
        try:
            new_grid = jorek_grid(id_generator()).read_grid_hdf5(filename)
            if DIAGNOSTIC_BASIS_SCALE:
                _diagnostic_nodes_xx_before = new_grid.nodes_xx.copy()
            report_grid_array_memory(new_grid)
            report_basis_scale_diagnostics(new_grid)
            report_memory("after HDF5 read")
            (
                new_scene, new_nodes, new_elements, new_boundaries,
            ) = build_grid_scene(
                new_grid, this_scaling,
                editable_depth=self.view.editable_depth,
            )
            verify_diagnostic_grid_unchanged(new_grid, "after construction")
        except Exception as error:
            (
                jorek, scene, view, node_list, element_list, boundary_list,
                old_view_scene,
            ) = old_state
            if old_view_scene is not None:
                self.view.setScene(old_view_scene)
            self._report_file_error("Could not open grid", error, interactive)
            return False

        jorek = new_grid
        scene = new_scene
        view = self.view
        node_list = new_nodes
        element_list = new_elements
        boundary_list = new_boundaries
        self.view.setScene(scene)
        self.view.rebuild_wall_outline_item()
        self.view.clear_geometry_undo()
        self.view._element_adjacency = getattr(
            scene, "_element_adjacency", None
        )
        if self.view._element_adjacency is None:
            self.view._element_adjacency = element_edge_adjacency(element_list)
        self.view.editable_element_indices_set = editable_element_indices(
            element_list, boundary_list,
            depth=self.view.editable_depth,
            adjacency=self.view._element_adjacency,
        )
        self.view.editable_node_indices_set = editable_node_indices(
            element_list, self.view.editable_element_indices_set
        )
        report_memory("after setScene")
        self.view.rubberBand = None
        self.view.start_point = None
        self.view.end_point = None
        self.view.rubberband_mode = None
        self.view.selected_point = None
        self.view.dragged_node = None
        self.view.selected_edges = []
        self.view.selected_nodes = []
        self.view.selected_elements = []
        self.view.current_patch = None
        self.view.current_extended_patch = None
        self.view.pending_main_uv_index = None
        self.view.pending_radial_layers = 1
        self.view.pending_bezier_mode = True
        self.current_filename = os.path.abspath(filename)
        self.source_is_grid_only = new_grid.grid_only_source
        self.view.document_modified = False
        self.view.set_patch_status("Grid opened")
        self.patch_controls.update_wall_label()
        self.view.update_patch_controls()
        self.view.fit_grid_to_window()
        report_memory("after fit")
        self.update_window_title()
        return True

    def save_grid_file(self, filename=None, interactive=False):
        destination = filename or self.current_filename
        if destination is None:
            if interactive:
                return self.save_grid_as_dialog()
            return False
        destination = os.path.abspath(destination)
        if (
            not self.source_is_grid_only
            and self.current_filename is not None
            and destination == os.path.abspath(self.current_filename)
        ):
            message = (
                "The original file contains simulation data. Use Save As "
                "to write the edited grid without overwriting the restart."
            )
            self._report_file_error("Save requires Save As", message, interactive)
            return False
        try:
            nodes_xx, boundary, vertices, element_sizes = live_grid_arrays()
            destination_directory = os.path.dirname(destination) or os.curdir
            os.makedirs(destination_directory, exist_ok=True)
            descriptor, temporary_filename = tempfile.mkstemp(
                prefix=".jorek_grid_", suffix=".h5",
                dir=destination_directory,
            )
            os.close(descriptor)
            try:
                globals()["jorek"].write_grid_hdf5(
                    temporary_filename, nodes_xx, boundary,
                    vertices, element_sizes,
                )
                os.replace(temporary_filename, destination)
            finally:
                if os.path.exists(temporary_filename):
                    os.remove(temporary_filename)
        except Exception as error:
            self._report_file_error("Could not save grid", error, interactive)
            return False
        self.current_filename = destination
        self.source_is_grid_only = True
        self.view.document_modified = False
        self.view.set_patch_status("Grid saved")
        self.update_window_title()
        return True

    def open_grid_dialog(self):
        if not self.maybe_save_changes():
            return False
        filename, unused_filter = QFileDialog.getOpenFileName(
            self, "Open JOREK grid", "jorek*.h5",
            "JOREK HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        return bool(filename) and self.open_grid_file(filename, interactive=True)

    def save_grid_from_menu(self):
        return self.save_grid_file(interactive=True)

    def save_grid_as_dialog(self):
        if self.current_filename:
            base, unused_extension = os.path.splitext(self.current_filename)
            suggested = base + "_edited.h5"
        else:
            suggested = "jorek_grid_edited.h5"
        filename, unused_filter = QFileDialog.getSaveFileName(
            self, "Save JOREK grid", suggested,
            "JOREK HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not filename:
            return False
        if not os.path.splitext(filename)[1]:
            filename += ".h5"
        return self.save_grid_file(filename, interactive=True)

    def maybe_save_changes(self):
        if not self.view.document_modified:
            return True
        choice = QMessageBox.warning(
            self, "Unsaved changes",
            "The grid has unsaved changes.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if choice == QMessageBox.Cancel:
            return False
        if choice == QMessageBox.Discard:
            return True
        return self.save_grid_file(interactive=True)

    def closeEvent(self, event):
        if self.maybe_save_changes():
            event.accept()
        else:
            event.ignore()


class big_patch_node(QGraphicsItem):
    def __init__(self,position):
        super().__init__()
        print("new patch node : ",position)
        self.position = position

class big_patch(QGraphicsPathItem):
    def __init__(self):
        super().__init__()
        self.setZValue(Z_PATCH_PREVIEW)
        print("big_patch : init")
        
        self.corner_nodes = []
        self.edges        = []
        for this_edge in view.selected_edges:
            self.edges.append(this_edge)
        this_path = QPainterPath()
        for this_edge in self.edges:         
            this_path.addPath(this_edge.path())
        if len(self.edges) == 3:
            ordered_nodes, unused_ordered_edges = ordered_three_edge_chain(
                self.edges
            )
            this_path.moveTo(ordered_nodes[0].position)
            for ordered_node in ordered_nodes[1:]:
                this_path.lineTo(ordered_node.position)
            this_path.lineTo(ordered_nodes[0].position)
        self.setPath(this_path)

    def add_corner(self,node):

        self.corner_nodes.append(node)
        
        if len(self.corner_nodes) > 1 and len(self.edges) == 1:
            corner_positions = [
                np_point(self.corner_nodes[0].position),
                np_point(self.corner_nodes[1].position),
            ]
            new_at_node0_position, new_at_node1_position = (
                pair_corners_to_edge_endpoints(
                    np_point(self.edges[0].nodes[0].position),
                    np_point(self.edges[0].nodes[1].position),
                    corner_positions[0],
                    corner_positions[1],
                )
            )
            if np.array_equal(new_at_node0_position, corner_positions[0]):
                new_at_node0, new_at_node1 = self.corner_nodes
            else:
                new_at_node1, new_at_node0 = self.corner_nodes
            self.corner_nodes = [new_at_node0, new_at_node1]
        elif len(self.corner_nodes) > 1:
            order = reorder_corners(np_point(self.edges[0].nodes[0].position),
                                    np_point(self.edges[0].nodes[1].position),
                                    np_point(self.corner_nodes[0].position),
                                    np_point(self.corner_nodes[1].position))
            if (order[2] > order[3]):
                tmp_node = self.corner_nodes[0]
                self.corner_nodes[0] = self.corner_nodes[1]
                self.corner_nodes[1] = tmp_node

        this_path = QPainterPath()
        for this_edge in self.edges:         
            this_path.addPath(this_edge.path())
        self.setPath(this_path)
        this_path.moveTo(self.edges[0].nodes[1].position)
        if len(self.edges) == 1 and len(self.corner_nodes) == 2:
            new_at_node0, new_at_node1 = self.corner_nodes
            this_path.lineTo(new_at_node1.position)
            this_path.lineTo(new_at_node0.position)
        elif len(self.edges) == 2 and len(self.corner_nodes) == 1:
            shared_node, outer_node0, outer_node1 = two_edge_corner_nodes(
                self.edges[0], self.edges[1]
            )
            new_node = self.corner_nodes[0]
            this_path.moveTo(outer_node0.position)
            this_path.lineTo(shared_node.position)
            this_path.lineTo(outer_node1.position)
            this_path.lineTo(new_node.position)
            this_path.lineTo(outer_node0.position)
        else:
            for this_node in self.corner_nodes:
                this_path.lineTo(this_node.position)
        this_path.lineTo(self.edges[0].nodes[0].position)
        self.setPath(this_path)
        self.update()

    def paint(self, painter: QPainter, option, widget=None): 
        for node in self.corner_nodes[::-1]:
            painter.drawEllipse(node.position.x() - 3, node.position.y() - 3, 6, 6)     
        painter.setPen(QPen(Qt.black, 1.))
        painter.drawPath(self.path())


def signed_edge_sizes(edge_nodes, uv_index):
    edge_sizes = np.zeros((2,2))
    edge_sizes[0,:] = 1.
    edge_sizes[1,0] = np.sign(np.inner(
        np_point(edge_nodes[1].position - edge_nodes[0].position),
        edge_nodes[0].xx[:,uv_index],
    ))
    edge_sizes[1,1] = np.sign(np.inner(
        np_point(edge_nodes[0].position - edge_nodes[1].position),
        edge_nodes[1].xx[:,uv_index],
    ))
    return edge_sizes


def inherited_transverse_endpoint_size(
    inner_edge, existing_node, new_outer_position,
):
    """Orient the inner owner's transverse scale into a new radial cell."""
    if not getattr(inner_edge, "active", True):
        raise ValueError("Inner boundary edge must be active")
    owner = element_by_index(inner_edge.element_index)
    if owner is None or not getattr(owner, "active", True):
        raise ValueError(
            "Inner boundary edge has no active owning element"
        )

    endpoint_matches = [
        endpoint for endpoint, node in enumerate(inner_edge.nodes)
        if node is existing_node or node.index == existing_node.index
    ]
    if len(endpoint_matches) != 1:
        raise ValueError(
            "Existing node must be exactly one endpoint of the inner edge"
        )
    endpoint = endpoint_matches[0]

    local_nodes_index = list(inner_edge.local_nodes_index or [])
    if len(local_nodes_index) == 2:
        local_vertex = local_nodes_index[endpoint]
        if (
            local_vertex < 0
            or local_vertex >= len(owner.vertices)
            or owner.vertices[local_vertex] != existing_node.index
        ):
            raise ValueError(
                "Inner edge endpoint metadata does not match its owning element"
            )
    else:
        owner_matches = [
            local_vertex
            for local_vertex, vertex in enumerate(owner.vertices)
            if vertex == existing_node.index
        ]
        if len(owner_matches) != 1:
            raise ValueError(
                "Existing node does not map uniquely to the inner owner"
            )
        local_vertex = owner_matches[0]

    if inner_edge.uv_index not in (1, 2):
        raise ValueError("Inner boundary edge uv_index must be 1 or 2")
    perp_uv = inner_edge.uv_index % 2 + 1
    old_size = float(owner.sizes[perp_uv, local_vertex])
    if not np.isfinite(old_size) or old_size == 0.0:
        raise ValueError(
            "Inner owner transverse size must be finite and nonzero"
        )

    raw_basis = np.asarray(existing_node.xx[:, perp_uv], dtype=float)
    if (
        not np.all(np.isfinite(raw_basis))
        or np.linalg.norm(raw_basis) == 0.0
    ):
        raise ValueError(
            "Existing node transverse basis must be finite and nonzero"
        )
    radial_direction = (
        np_point(new_outer_position) - np_point(existing_node.position)
    )
    if (
        not np.all(np.isfinite(radial_direction))
        or np.linalg.norm(radial_direction) == 0.0
    ):
        raise ValueError("New transverse edge direction must be finite and nonzero")
    orientation = float(np.inner(raw_basis, radial_direction))
    if not np.isfinite(orientation) or orientation == 0.0:
        raise ValueError(
            "Existing transverse basis cannot be oriented toward the new node"
        )
    return math.copysign(abs(old_size), orientation)


def bezier_nodal_parameter_scales(parameters):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 1 or len(parameters) < 2:
        raise ValueError("Bezier parameters must contain at least two values")
    parameter_intervals = np.diff(parameters)
    if (
        not np.all(np.isfinite(parameter_intervals))
        or np.any(parameter_intervals <= 0.0)
    ):
        raise ValueError("Bezier parameter intervals must be positive and nonzero")

    scales = np.empty(len(parameters), dtype=float)
    scales[0] = parameter_intervals[0]
    scales[-1] = parameter_intervals[-1]
    if len(parameters) > 2:
        scales[1:-1] = 0.5 * (
            parameter_intervals[:-1] + parameter_intervals[1:]
        )
    return scales


def prescribed_bezier_outer_edge_sizes(
    edge_nodes, parameter_start_node, parameter_end_node, parameter_interval,
    parameter_start_scale, parameter_end_scale,
):
    """Use parameter ordering, never chord direction, for Bézier edge signs."""
    if parameter_interval <= 0.0:
        raise ValueError("Bezier parameter interval must be positive and nonzero")
    if parameter_start_scale <= 0.0 or parameter_end_scale <= 0.0:
        raise ValueError("Bezier nodal parameter scales must be positive and nonzero")
    start_size = parameter_interval / parameter_start_scale
    end_size = parameter_interval / parameter_end_scale
    edge_sizes = np.ones((2, 2))
    if (
        edge_nodes[0] is parameter_start_node
        and edge_nodes[1] is parameter_end_node
    ):
        edge_sizes[1, :] = [start_size, -end_size]
    elif (
        edge_nodes[0] is parameter_end_node
        and edge_nodes[1] is parameter_start_node
    ):
        edge_sizes[1, :] = [-end_size, start_size]
    else:
        raise ValueError("Bézier outer edge nodes do not match parameter order")
    return edge_sizes


def basis_size_for_effective_vector(stored_basis, effective_vector):
    """Return the scalar that makes a stored basis equal an effective vector."""
    stored_basis = np.asarray(stored_basis, dtype=float)
    effective_vector = np.asarray(effective_vector, dtype=float)
    basis_norm_squared = np.inner(stored_basis, stored_basis)
    if not np.isfinite(basis_norm_squared) or basis_norm_squared <= 0.0:
        raise ValueError("Bezier fixed-node basis vector must be nonzero")
    size = np.inner(stored_basis, effective_vector) / basis_norm_squared
    if (
        not np.isfinite(size)
        or not np.allclose(
            stored_basis * size, effective_vector, rtol=1.e-9, atol=1.e-12
        )
    ):
        raise ValueError(
            "Bezier fixed-node basis vector is not collinear with the tangent"
        )
    return size


def override_fixed_bezier_endpoint_sizes(
    edge_sizes, edge_nodes, parameter_start_node, parameter_end_node,
    parameter_interval, start_tangent, end_tangent, uv_index, fixed_nodes,
    fixed_effective_vectors=None,
):
    """Use each fixed node's actual stored basis for its endpoint multiplier."""
    intended_by_node = {
        parameter_start_node: parameter_interval * np.asarray(start_tangent) / 3.,
        parameter_end_node: -parameter_interval * np.asarray(end_tangent) / 3.,
    }
    for fixed_node in fixed_nodes or []:
        if fixed_node not in intended_by_node:
            raise ValueError("Bezier fixed node is not an edge endpoint")
        endpoint = next(
            index for index, node in enumerate(edge_nodes) if node is fixed_node
        )
        intended_vector = (fixed_effective_vectors or {}).get(
            fixed_node.index, intended_by_node[fixed_node]
        )
        edge_sizes[1, endpoint] = basis_size_for_effective_vector(
            fixed_node.xx[:, uv_index], intended_vector
        )
    return edge_sizes


def override_fixed_endpoint_sizes(
    edge_sizes, edge_nodes, uv_index, fixed_effective_vectors,
):
    """Encode prescribed effective vectors through existing raw nodal bases."""
    for endpoint, node in enumerate(edge_nodes):
        effective_vector = (fixed_effective_vectors or {}).get(node.index)
        if effective_vector is not None:
            edge_sizes[1, endpoint] = basis_size_for_effective_vector(
                node.xx[:, uv_index], effective_vector
            )
    return edge_sizes


def print_bezier_outer_edge_diagnostic(edge, patch):
    index = patch.along_edge_index
    parameter_interval = (
        patch.along_parameters[index + 1] - patch.along_parameters[index]
    )
    effective_vectors = [
        node.xx[:, edge.uv_index] * edge.sizes[1, endpoint]
        for endpoint, node in enumerate(edge.nodes)
    ]
    intended_vectors = [
        parameter_interval * np.asarray(patch.along_tangents[index]) / 3.0,
        -parameter_interval
        * np.asarray(patch.along_tangents[index + 1]) / 3.0,
    ]
    print(
        "Bezier outer edge",
        index,
        [node.index for node in edge.nodes],
        "parameter order",
        patch.along_parameters[index],
        patch.along_parameters[index + 1],
        "sizes",
        edge.sizes[1, :],
        "effective stored order",
        effective_vectors,
        "intended parameter order",
        intended_vectors,
    )


def orient_new_node_red_vector(
    new_node, reference_node, column, main_uv_index, perp_index
):
    """Orient the displayed red vector before any new edge is constructed."""
    if reference_node is None:
        return
    print("main edge uv_index =", main_uv_index)
    print("perp_index =", perp_index)
    print("column", column)
    print("reference red =", reference_node.xx[:, 2])
    print("new red before =", new_node.xx[:, 2])
    dot_before = np.dot(reference_node.xx[:, 2], new_node.xx[:, 2])
    print("dot before =", dot_before)
    if dot_before < 0:
        new_node.xx[:, 2] *= -1.
        jorek.nodes_xx[:, 2, new_node.index] *= -1.
        new_node.red_handle.sync_position()
    print("new red after =", new_node.xx[:, 2])
    print(
        "dot after =",
        np.dot(reference_node.xx[:, 2], new_node.xx[:, 2]),
    )


class single_element_patch_data:
    def __init__(
        self, edges, corner_nodes, radial_reference_nodes=None,
        radial_reference_node=None, radial_columns=None,
        along_tangents=None, along_tangent=None, along_parameter_interval=None,
        along_edge_index=None, along_parameters=None, fixed_bezier_nodes=None,
        parameter_start_node=None, parameter_end_node=None,
        along_basis_vectors=None, along_basis_vector=None,
        fixed_bezier_effective_vectors=None,
    ):
        self.edges = list(edges)
        self.corner_nodes = list(corner_nodes)
        self.radial_reference_nodes = radial_reference_nodes
        self.radial_reference_node = radial_reference_node
        self.radial_columns = radial_columns
        self.along_tangents = along_tangents
        self.along_tangent = along_tangent
        self.along_parameter_interval = along_parameter_interval
        self.along_edge_index = along_edge_index
        self.along_parameters = along_parameters
        self.fixed_bezier_nodes = list(fixed_bezier_nodes or [])
        self.parameter_start_node = parameter_start_node
        self.parameter_end_node = parameter_end_node
        self.along_basis_vectors = along_basis_vectors
        self.along_basis_vector = along_basis_vector
        self.fixed_bezier_effective_vectors = fixed_bezier_effective_vectors


def preview_node(point_or_node):
    if hasattr(point_or_node, "position"):
        return point_or_node
    return big_patch_node(point_or_node)


def add_element_from_one_edge(
    edge, point0, point1, radial_reference_nodes=None, radial_columns=None,
    along_tangents=None, along_parameter_interval=None,
    along_edge_index=None, along_parameters=None, along_basis_vectors=None,
):
    patch = single_element_patch_data(
        [edge], [preview_node(point0), preview_node(point1)],
        radial_reference_nodes=radial_reference_nodes,
        radial_columns=radial_columns,
        along_tangents=along_tangents,
        along_parameter_interval=along_parameter_interval,
        along_edge_index=along_edge_index,
        along_parameters=along_parameters,
        along_basis_vectors=along_basis_vectors,
    )
    element_count = len(element_list)
    _add_single_element_patch_to_nodes_elements(patch)
    return element_list[-1] if len(element_list) > element_count else None


def add_element_from_two_edges(
    edge0, edge1, point, radial_reference_node=None, radial_column=None,
    along_tangent=None, along_tangents=None, along_parameter_interval=None,
    along_edge_index=None, along_parameters=None, fixed_bezier_nodes=None,
    along_basis_vector=None,
    fixed_bezier_effective_vectors=None,
):
    patch = single_element_patch_data(
        [edge0, edge1], [preview_node(point)],
        radial_reference_node=radial_reference_node,
        radial_columns=([radial_column] if radial_column is not None else None),
        along_tangents=along_tangents,
        along_tangent=along_tangent,
        along_parameter_interval=along_parameter_interval,
        along_edge_index=along_edge_index,
        along_parameters=along_parameters,
        fixed_bezier_nodes=fixed_bezier_nodes,
        along_basis_vector=along_basis_vector,
        fixed_bezier_effective_vectors=fixed_bezier_effective_vectors,
    )
    element_count = len(element_list)
    _add_single_element_patch_to_nodes_elements(patch)
    return element_list[-1] if len(element_list) > element_count else None


def add_element_from_three_edges(
    edge0, edge1, edge2, along_tangents=None, along_parameters=None,
    along_edge_index=None, parameter_start_node=None,
    parameter_end_node=None, fixed_bezier_nodes=None,
    fixed_bezier_effective_vectors=None,
):
    along_parameter_interval = (
        along_parameters[along_edge_index + 1]
        - along_parameters[along_edge_index]
        if along_parameters is not None else None
    )
    patch = single_element_patch_data(
        [edge0, edge1, edge2], [],
        along_tangents=along_tangents,
        along_parameter_interval=along_parameter_interval,
        along_edge_index=along_edge_index,
        along_parameters=along_parameters,
        fixed_bezier_nodes=fixed_bezier_nodes,
        parameter_start_node=parameter_start_node,
        parameter_end_node=parameter_end_node,
        fixed_bezier_effective_vectors=fixed_bezier_effective_vectors,
    )
    element_count = len(element_list)
    _add_single_element_patch_to_nodes_elements(patch)
    return element_list[-1] if len(element_list) > element_count else None


def add_patch_to_nodes_elements(patch):
    if len(patch.edges) == 1 and len(patch.corner_nodes) == 2:
        return add_element_from_one_edge(
            patch.edges[0], patch.corner_nodes[0], patch.corner_nodes[1]
        )
    if len(patch.edges) == 2 and len(patch.corner_nodes) == 1:
        return add_element_from_two_edges(
            patch.edges[0], patch.edges[1], patch.corner_nodes[0]
        )
    if len(patch.edges) == 3 and not patch.corner_nodes:
        return add_element_from_three_edges(*patch.edges)
    return _add_single_element_patch_to_nodes_elements(patch)


def find_boundary_edge_to_position(inner_node, outer_position):
    matches = []
    for edge in boundary_list:
        if inner_node.index not in edge.vertices:
            continue
        other_node = next(
            node for node in edge.nodes if node.index != inner_node.index
        )
        if np.allclose(np_point(other_node.position), np_point(outer_position)):
            matches.append(edge)
    if len(matches) != 1:
        raise ValueError(
            "Could not identify the newly created transverse boundary edge"
        )
    return matches[0]


def find_boundary_edge_between_nodes(node0, node1):
    node_indices = frozenset((node0.index, node1.index))
    matches = [
        edge for edge in boundary_list
        if getattr(edge, "active", True)
        and frozenset(edge.vertices) == node_indices
    ]
    if len(matches) != 1:
        raise ValueError("Could not identify the created outer boundary edge")
    return matches[0]


def add_open_extended_row(
    inner_nodes, inner_edges, target_positions,
    reference_inner_nodes=None, radial_row=None,
    along_tangents=None, along_parameters=None, along_basis_vectors=None,
):
    """Create one radial row using the proven sequential element helpers."""
    first_new_element_index = len(element_list)
    if add_element_from_one_edge(
        inner_edges[0], target_positions[0], target_positions[1],
        radial_reference_nodes=(
            reference_inner_nodes[:2]
            if reference_inner_nodes is not None else None
        ),
        radial_columns=([0, 1] if reference_inner_nodes is not None else None),
        along_tangents=(along_tangents[:2] if along_tangents is not None else None),
        along_parameter_interval=(
            along_parameters[1] - along_parameters[0]
            if along_parameters is not None else None
        ),
        along_edge_index=(0 if along_parameters is not None else None),
        along_parameters=along_parameters,
        along_basis_vectors=(
            along_basis_vectors[:2]
            if along_basis_vectors is not None else None
        ),
    ) is None:
        return None
    for index in range(1, len(inner_edges)):
        transverse_edge = find_boundary_edge_to_position(
            inner_nodes[index], target_positions[index]
        )
        if add_element_from_two_edges(
            inner_edges[index], transverse_edge, target_positions[index + 1],
            radial_reference_node=(
                reference_inner_nodes[index + 1]
                if reference_inner_nodes is not None else None
            ),
            radial_column=(index + 1 if reference_inner_nodes is not None else None),
            along_tangent=(
                along_tangents[index + 1]
                if along_tangents is not None else None
            ),
            along_tangents=along_tangents,
            along_parameter_interval=(
                along_parameters[index + 1] - along_parameters[index]
                if along_parameters is not None else None
            ),
            along_edge_index=(index if along_parameters is not None else None),
            along_parameters=along_parameters,
            along_basis_vector=(
                along_basis_vectors[index + 1]
                if along_basis_vectors is not None else None
            ),
        ) is None:
            return None

    created_elements = element_list[first_new_element_index:]
    inner_indices = {node.index for node in inner_nodes}
    created_outer_indices = {
        vertex
        for element in created_elements
        for vertex in element.vertices
        if vertex not in inner_indices
    }
    created_outer_nodes = [node_list[index] for index in created_outer_indices]
    outer_nodes = []
    for target_position in target_positions:
        matches = [
            node for node in created_outer_nodes
            if np.allclose(np_point(node.position), np_point(target_position))
        ]
        if len(matches) != 1:
            raise ValueError("Could not identify a created outer node")
        outer_nodes.append(matches[0])
    outer_edges = [
        find_boundary_edge_between_nodes(outer_nodes[index], outer_nodes[index + 1])
        for index in range(len(outer_nodes) - 1)
    ]
    return outer_nodes, outer_edges, created_elements


def add_extended_patch_to_nodes_elements(patch):
    if patch.capped_gap is not None:
        return add_capped_gap_to_nodes_elements(patch)
    if patch.one_cap_topology is not None:
        return add_one_cap_gap_to_nodes_elements(patch)
    if not patch.can_commit:
        print("Element creation for capped boundary gaps is not implemented yet")
        return
    if len(patch.outer_nodes) != patch.required_outer_node_count:
        print(
            "Extended patch requires exactly",
            patch.required_outer_node_count,
            "outer points",
        )
        return

    if len(patch.preview_node_rows) != patch.radial_layers + 1:
        print("Extended patch radial preview rows are incomplete")
        return

    inner_nodes = list(patch.ordered_nodes)
    reference_inner_nodes = list(patch.ordered_nodes)
    inner_edges = list(patch.ordered_edges)
    first_new_element_index = len(element_list)
    node_rows = [inner_nodes]
    for radial_index in range(patch.radial_layers):
        target_positions = [
            node.position
            for node in patch.preview_node_rows[radial_index + 1]
        ]
        row_result = add_open_extended_row(
            inner_nodes, inner_edges, target_positions,
            reference_inner_nodes=(
                reference_inner_nodes if patch.radial_layers > 1 else None
            ),
            radial_row=radial_index + 1,
            along_tangents=(
                patch.outer_tangents
                if patch.bezier_mode
                and radial_index == patch.radial_layers - 1
                else None
            ),
            along_parameters=(
                patch.outer_parameters
                if patch.bezier_mode
                and radial_index == patch.radial_layers - 1
                else None
            ),
            along_basis_vectors=(
                patch.preview_along_vectors[radial_index + 1]
                if patch.bezier_mode else None
            ),
        )
        if row_result is None:
            return
        next_nodes, next_edges, unused_created_elements = row_result
        inner_nodes, inner_edges = next_nodes, next_edges
        node_rows.append(inner_nodes)

    created_elements = element_list[first_new_element_index:]
    recompute_node_boundaries(node_list, boundary_list)
    rebuild_node_connections()
    if patch.scene() is scene:
        scene.removeItem(patch)
    view.current_extended_patch = None
    view.selected_edges = []
    return node_rows, created_elements


def two_cap_topology_error(topology, ordered_nodes):
    start_cap_edges = list(topology.start_cap_edges)
    end_cap_edges = list(topology.end_cap_edges)
    start_cap_nodes = list(topology.start_cap_nodes)
    end_cap_nodes = list(topology.end_cap_nodes)
    if not start_cap_edges or not end_cap_edges:
        return "Two-cap extended patch requires exactly two cap chains"
    if len(start_cap_edges) != len(end_cap_edges):
        return "Two-cap extended patch requires equal cap-chain lengths"
    radial_layers = len(start_cap_edges)
    if len(start_cap_nodes) != radial_layers + 1:
        return "Two-cap start node count does not match its cap chain"
    if len(end_cap_nodes) != radial_layers + 1:
        return "Two-cap end node count does not match its cap chain"
    if start_cap_nodes[0] is not ordered_nodes[0]:
        return "Two-cap start chain is not ordered from the main-row endpoint"
    if end_cap_nodes[0] is not ordered_nodes[-1]:
        return "Two-cap end chain is not ordered from the main-row endpoint"
    for radial_index in range(radial_layers):
        if frozenset(start_cap_edges[radial_index].vertices) != frozenset((
            start_cap_nodes[radial_index].index,
            start_cap_nodes[radial_index + 1].index,
        )):
            return "Two-cap start edge does not connect consecutive cap nodes"
        if frozenset(end_cap_edges[radial_index].vertices) != frozenset((
            end_cap_nodes[radial_index].index,
            end_cap_nodes[radial_index + 1].index,
        )):
            return "Two-cap end edge does not connect consecutive cap nodes"
    return None


def add_two_cap_extended_row(
    inner_nodes, inner_edges, target_positions,
    start_cap_edge, end_cap_edge, fixed_start_node, fixed_end_node,
    reference_inner_nodes=None, along_tangents=None, along_parameters=None,
    along_basis_vectors=None,
):
    """Create one radial row whose two endpoints are existing cap nodes."""
    if len(inner_nodes) != len(inner_edges) + 1:
        raise ValueError("Two-cap inner row has inconsistent node/edge counts")
    if len(target_positions) != len(inner_nodes):
        raise ValueError("Two-cap target row has the wrong number of positions")
    if frozenset(start_cap_edge.vertices) != frozenset((
        inner_nodes[0].index, fixed_start_node.index,
    )):
        raise ValueError("Two-cap start edge connects unexpected nodes")
    if frozenset(end_cap_edge.vertices) != frozenset((
        inner_nodes[-1].index, fixed_end_node.index,
    )):
        raise ValueError("Two-cap end edge connects unexpected nodes")
    if not np.allclose(
        np_point(target_positions[0]), np_point(fixed_start_node.position)
    ):
        raise ValueError("Two-cap target row has the wrong fixed start node")
    if not np.allclose(
        np_point(target_positions[-1]), np_point(fixed_end_node.position)
    ):
        raise ValueError("Two-cap target row has the wrong fixed end node")
    if reference_inner_nodes is not None and (
        len(reference_inner_nodes) != len(inner_nodes)
    ):
        raise ValueError("Two-cap radial reference row has the wrong node count")
    if (along_tangents is None) != (along_parameters is None):
        raise ValueError("Two-cap Bezier tangents and parameters must be paired")
    if along_parameters is not None:
        if (
            len(along_parameters) != len(inner_nodes)
            or len(along_tangents) != len(inner_nodes)
        ):
            raise ValueError("Two-cap Bezier samples have the wrong count")
        bezier_nodal_parameter_scales(along_parameters)
    if along_basis_vectors is not None and (
        len(along_basis_vectors) != len(inner_nodes)
    ):
        raise ValueError("Two-cap along-vector row has the wrong count")

    first_new_node_index = len(node_list)
    first_new_element_index = len(element_list)
    next_nodes = [fixed_start_node]
    if len(inner_edges) == 1:
        bezier_kwargs = {}
        if along_parameters is not None:
            bezier_kwargs = {
                "along_tangents": along_tangents,
                "along_parameters": along_parameters,
                "along_edge_index": 0,
                "parameter_start_node": fixed_start_node,
                "parameter_end_node": fixed_end_node,
                "fixed_bezier_nodes": [fixed_start_node, fixed_end_node],
            }
        if along_basis_vectors is not None:
            bezier_kwargs["fixed_bezier_effective_vectors"] = {
                fixed_start_node.index: along_basis_vectors[0],
                fixed_end_node.index: -np.asarray(along_basis_vectors[-1]),
            }
        if add_element_from_three_edges(
            inner_edges[0], start_cap_edge, end_cap_edge,
            **bezier_kwargs
        ) is None:
            return None
    else:
        transverse_edge = start_cap_edge
        for column in range(len(inner_edges) - 1):
            old_node_count = len(node_list)
            radial_kwargs = {}
            if reference_inner_nodes is not None:
                radial_kwargs = {
                    "radial_reference_node": reference_inner_nodes[column + 1],
                    "radial_column": column + 1,
                }
            bezier_kwargs = {}
            if along_basis_vectors is not None:
                bezier_kwargs["along_basis_vector"] = (
                    along_basis_vectors[column + 1]
                )
            if along_parameters is not None:
                bezier_kwargs.update({
                    "along_tangent": along_tangents[column + 1],
                    "along_tangents": along_tangents,
                    "along_parameter_interval": (
                        along_parameters[column + 1]
                        - along_parameters[column]
                    ),
                    "along_edge_index": column,
                    "along_parameters": along_parameters,
                    "fixed_bezier_nodes": (
                        [fixed_start_node] if column == 0 else None
                    ),
                })
            if along_basis_vectors is not None and column == 0:
                bezier_kwargs["fixed_bezier_effective_vectors"] = {
                    fixed_start_node.index: along_basis_vectors[0]
                }
            if add_element_from_two_edges(
                inner_edges[column], transverse_edge,
                target_positions[column + 1],
                **radial_kwargs, **bezier_kwargs
            ) is None:
                return None
            if len(node_list) != old_node_count + 1:
                raise ValueError(
                    "Two-cap row did not create exactly one interior node"
                )
            next_node = node_list[-1]
            if not np.allclose(
                np_point(next_node.position),
                np_point(target_positions[column + 1]),
            ):
                raise ValueError(
                    "Two-cap interior node does not match its target position"
                )
            next_nodes.append(next_node)
            transverse_edge = find_boundary_edge_between_nodes(
                inner_nodes[column + 1], next_node
            )
        old_node_count = len(node_list)
        bezier_kwargs = {}
        if along_parameters is not None:
            bezier_kwargs = {
                "along_tangents": along_tangents,
                "along_parameters": along_parameters,
                "along_edge_index": len(inner_edges) - 1,
                "parameter_start_node": next_nodes[-1],
                "parameter_end_node": fixed_end_node,
                "fixed_bezier_nodes": [fixed_end_node],
            }
        if along_basis_vectors is not None:
            bezier_kwargs["fixed_bezier_effective_vectors"] = {
                fixed_end_node.index: -np.asarray(along_basis_vectors[-1])
            }
        if add_element_from_three_edges(
            inner_edges[-1], transverse_edge, end_cap_edge,
            **bezier_kwargs
        ) is None:
            return None
        if len(node_list) != old_node_count:
            raise ValueError("Two-cap row closure unexpectedly created a node")

    if len(node_list) != first_new_node_index + max(len(inner_edges) - 1, 0):
        raise ValueError("Two-cap row created an unexpected number of nodes")
    next_nodes.append(fixed_end_node)
    next_edges = [
        find_boundary_edge_between_nodes(next_nodes[index], next_nodes[index + 1])
        for index in range(len(next_nodes) - 1)
    ]
    created_elements = element_list[first_new_element_index:]
    if len(created_elements) != len(inner_edges):
        raise ValueError("Two-cap row created an unexpected number of elements")
    return next_nodes, next_edges, created_elements


def add_capped_gap_to_nodes_elements(patch):
    gap = patch.capped_gap
    topology_error = two_cap_topology_error(gap, patch.ordered_nodes)
    if topology_error:
        print(topology_error)
        return
    start_cap_edges = list(gap.start_cap_edges)
    end_cap_edges = list(gap.end_cap_edges)
    start_cap_nodes = list(gap.start_cap_nodes)
    end_cap_nodes = list(gap.end_cap_nodes)
    if len(patch.outer_nodes) != patch.required_outer_node_count:
        print(
            "Two-cap extended patch requires exactly",
            patch.required_outer_node_count,
            "outer preview nodes before commit",
        )
        return
    if len(start_cap_edges) != patch.radial_layers:
        print("Two-cap edge count does not match radial layer count")
        return
    if len(patch.preview_node_rows) != patch.radial_layers + 1:
        print("Two-cap radial preview rows are incomplete")
        return
    if any(
        len(row) != patch.required_outer_node_count
        for row in patch.preview_node_rows
    ):
        print("Two-cap radial preview rows are incomplete")
        return
    for radial_index, row in enumerate(patch.preview_node_rows):
        if row[0] is not start_cap_nodes[radial_index]:
            print("Two-cap preview row does not reuse its start cap node")
            return
        if row[-1] is not end_cap_nodes[radial_index]:
            print("Two-cap preview row does not reuse its end cap node")
            return
    working_tangents = working_parameters = None
    if patch.bezier_mode:
        working_tangents = [
            np.asarray(tangent, dtype=float)
            for tangent in patch.outer_tangents
        ]
        working_parameters = np.asarray(
            patch.outer_parameters, dtype=float
        )
        try:
            if (
                len(working_parameters) != patch.required_outer_node_count
                or len(working_tangents) != patch.required_outer_node_count
            ):
                raise ValueError(
                    "Two-cap Bezier samples do not match the outer nodes"
                )
            bezier_nodal_parameter_scales(working_parameters)
            main_uv_index = patch.main_uv_index()
            basis_size_for_effective_vector(
                start_cap_nodes[-1].xx[:, main_uv_index],
                (working_parameters[1] - working_parameters[0])
                * working_tangents[0] / 3.,
            )
            basis_size_for_effective_vector(
                end_cap_nodes[-1].xx[:, main_uv_index],
                -(working_parameters[-1] - working_parameters[-2])
                * working_tangents[-1] / 3.,
            )
        except ValueError as error:
            print(error)
            return

    inner_nodes = list(patch.ordered_nodes)
    inner_edges = list(patch.ordered_edges)
    reference_inner_nodes = list(patch.ordered_nodes)
    first_new_element_index = len(element_list)
    node_rows = [inner_nodes]
    for radial_index in range(patch.radial_layers):
        if inner_nodes[0] is not start_cap_nodes[radial_index]:
            raise ValueError("Two-cap row has the wrong current start node")
        if inner_nodes[-1] is not end_cap_nodes[radial_index]:
            raise ValueError("Two-cap row has the wrong current end node")
        target_positions = [
            node.position for node in patch.preview_node_rows[radial_index + 1]
        ]
        row_result = add_two_cap_extended_row(
            inner_nodes, inner_edges, target_positions,
            start_cap_edges[radial_index], end_cap_edges[radial_index],
            start_cap_nodes[radial_index + 1],
            end_cap_nodes[radial_index + 1],
            reference_inner_nodes=(
                reference_inner_nodes if patch.radial_layers > 1 else None
            ),
            along_tangents=(
                working_tangents
                if radial_index == patch.radial_layers - 1 else None
            ),
            along_parameters=(
                working_parameters
                if radial_index == patch.radial_layers - 1 else None
            ),
            along_basis_vectors=(
                patch.preview_along_vectors[radial_index + 1]
                if patch.bezier_mode else None
            ),
        )
        if row_result is None:
            return
        next_nodes, next_edges, unused_created_elements = row_result
        inner_nodes, inner_edges = next_nodes, next_edges
        node_rows.append(inner_nodes)

    created_elements = element_list[first_new_element_index:]
    recompute_node_boundaries(node_list, boundary_list)
    rebuild_node_connections()
    if patch.scene() is scene:
        scene.removeItem(patch)
    view.current_extended_patch = None
    view.selected_edges = []
    return node_rows, created_elements


def cap_first_rows(topology, preview_node_rows):
    inner_nodes = list(topology.inner_nodes)
    inner_edges = list(topology.inner_edges)
    preview_node_rows = [list(row) for row in preview_node_rows]
    if topology.start_cap_edges:
        return (
            inner_nodes, inner_edges, preview_node_rows,
            list(topology.start_cap_nodes), list(topology.start_cap_edges),
        )
    return (
        list(reversed(inner_nodes)),
        list(reversed(inner_edges)),
        [list(reversed(row)) for row in preview_node_rows],
        list(topology.end_cap_nodes), list(topology.end_cap_edges),
    )


def cap_first_bezier_data(patch, topology):
    """Return strictly increasing Bezier samples in cap-first row order."""
    parameters = np.asarray(patch.outer_parameters, dtype=float)
    tangents = [np.asarray(tangent, dtype=float) for tangent in patch.outer_tangents]
    if len(parameters) != len(patch.outer_nodes) or len(tangents) != len(parameters):
        raise ValueError("One-cap Bezier samples do not match the outer nodes")
    if not topology.start_cap_edges:
        parameters = 1.0 - parameters[::-1]
        tangents = [-tangent for tangent in reversed(tangents)]
    bezier_nodal_parameter_scales(parameters)
    return tangents, parameters


def cap_first_global_bezier_row_data(patch, topology, radial_index):
    """Return one exact global row in the commit path's cap-first order."""
    data = patch.one_cap_global_rows[radial_index]
    parameters = np.asarray(data["parameters"], dtype=float)
    tangents = [np.asarray(tangent, dtype=float) for tangent in data["tangents"]]
    segments = [np.asarray(segment, dtype=float) for segment in data["segments"]]
    if not topology.start_cap_edges:
        parameters = 1.0 - parameters[::-1]
        tangents = [-tangent for tangent in reversed(tangents)]
        segments = [segment[::-1] for segment in reversed(segments)]
    bezier_nodal_parameter_scales(parameters)
    return tangents, parameters, segments


def add_one_cap_extended_row(
    inner_nodes, inner_edges, target_positions, cap_edge, fixed_outer_node,
    reference_inner_nodes=None, along_tangents=None, along_parameters=None,
    along_basis_vectors=None, fixed_cap_effective_vector=None,
):
    """Create one cap-first radial row without duplicating its fixed node."""
    if len(inner_nodes) != len(inner_edges) + 1:
        raise ValueError("One-cap inner row has inconsistent node/edge counts")
    if len(target_positions) != len(inner_nodes):
        raise ValueError("One-cap target row has the wrong number of positions")
    if inner_nodes[0].index not in cap_edge.vertices:
        raise ValueError("One-cap edge is not attached to the inner-row endpoint")
    if fixed_outer_node.index not in cap_edge.vertices:
        raise ValueError("One-cap edge does not contain the fixed outer node")
    if frozenset(cap_edge.vertices) != frozenset((
        inner_nodes[0].index, fixed_outer_node.index,
    )):
        raise ValueError("One-cap edge connects unexpected nodes")
    if not np.allclose(
        np_point(target_positions[0]), np_point(fixed_outer_node.position)
    ):
        raise ValueError("One-cap target row does not start at its fixed cap node")
    if reference_inner_nodes is not None and (
        len(reference_inner_nodes) != len(inner_nodes)
    ):
        raise ValueError("One-cap radial reference row has the wrong node count")
    if (along_tangents is None) != (along_parameters is None):
        raise ValueError("One-cap Bezier tangents and parameters must be paired")
    if along_parameters is not None:
        if (
            len(along_parameters) != len(inner_nodes)
            or len(along_tangents) != len(inner_nodes)
        ):
            raise ValueError("One-cap Bezier samples have the wrong count")
        bezier_nodal_parameter_scales(along_parameters)
    if along_basis_vectors is not None and (
        len(along_basis_vectors) != len(inner_nodes)
    ):
        raise ValueError("One-cap along-vector row has the wrong count")

    first_new_node_index = len(node_list)
    first_new_element_index = len(element_list)
    left_transverse_edge = cap_edge
    for index, inner_edge in enumerate(inner_edges):
        if add_element_from_two_edges(
            inner_edge, left_transverse_edge, target_positions[index + 1],
            radial_reference_node=(
                reference_inner_nodes[index + 1]
                if reference_inner_nodes is not None else None
            ),
            radial_column=(
                index + 1 if reference_inner_nodes is not None else None
            ),
            along_tangent=(
                along_tangents[index + 1]
                if along_tangents is not None else None
            ),
            along_tangents=along_tangents,
            along_parameter_interval=(
                along_parameters[index + 1] - along_parameters[index]
                if along_parameters is not None else None
            ),
            along_edge_index=(index if along_parameters is not None else None),
            along_parameters=along_parameters,
            fixed_bezier_nodes=(
                [fixed_outer_node]
                if along_parameters is not None and index == 0 else None
            ),
            along_basis_vector=(
                along_basis_vectors[index + 1]
                if along_basis_vectors is not None else None
            ),
            fixed_bezier_effective_vectors=(
                {fixed_outer_node.index: fixed_cap_effective_vector}
                if fixed_cap_effective_vector is not None and index == 0
                else None
            ),
        ) is None:
            return None
        if index < len(inner_edges) - 1:
            left_transverse_edge = find_boundary_edge_to_position(
                inner_nodes[index + 1], target_positions[index + 1]
            )

    new_nodes = list(node_list[first_new_node_index:])
    if len(new_nodes) != len(inner_edges):
        raise ValueError("One-cap row created an unexpected number of nodes")
    next_nodes = [fixed_outer_node] + new_nodes
    for index, (node, target_position) in enumerate(zip(
        next_nodes, target_positions
    )):
        if not np.allclose(np_point(node.position), np_point(target_position)):
            raise ValueError(
                "One-cap row node {} does not match its target position".format(
                    index
                )
            )
    next_edges = [
        find_boundary_edge_between_nodes(next_nodes[index], next_nodes[index + 1])
        for index in range(len(next_nodes) - 1)
    ]
    created_elements = element_list[first_new_element_index:]
    return next_nodes, next_edges, created_elements


def add_one_cap_gap_to_nodes_elements(patch):
    if len(patch.outer_nodes) != patch.required_outer_node_count:
        print(
            "One-cap extended patch requires exactly",
            patch.required_outer_node_count,
            "outer preview nodes before commit",
        )
        return
    topology = patch.one_cap_topology
    start_cap_edges = list(topology.start_cap_edges)
    end_cap_edges = list(topology.end_cap_edges)
    if bool(start_cap_edges) == bool(end_cap_edges):
        print("One-cap extended patch requires exactly one cap chain")
        return
    cap_edges = start_cap_edges or end_cap_edges
    cap_nodes = list(
        topology.start_cap_nodes if start_cap_edges else topology.end_cap_nodes
    )
    if len(cap_edges) != patch.radial_layers:
        print("One-cap edge count does not match radial layer count")
        return
    if len(cap_nodes) != patch.radial_layers + 1:
        print("One-cap node count does not match radial layer count")
        return
    if len(patch.preview_node_rows) != patch.radial_layers + 1:
        print("One-cap radial preview rows are incomplete")
        return
    if any(
        len(row) != patch.required_outer_node_count
        for row in patch.preview_node_rows
    ):
        print("One-cap radial preview rows are incomplete")
        return

    (
        inner_nodes, inner_edges, preview_node_rows, cap_nodes, cap_edges,
    ) = cap_first_rows(
        topology, patch.preview_node_rows
    )
    working_vector_rows = None
    if patch.bezier_mode:
        working_vector_rows = [
            list(row) for row in patch.preview_along_vectors
        ]
        if not topology.start_cap_edges:
            working_vector_rows = [
                list(reversed(row)) for row in working_vector_rows
            ]
    working_tangents = working_parameters = None
    if patch.bezier_mode:
        try:
            working_tangents, working_parameters = cap_first_bezier_data(
                patch, topology
            )
            basis_size_for_effective_vector(
                cap_nodes[-1].xx[:, patch.main_uv_index()],
                (working_parameters[1] - working_parameters[0])
                * working_tangents[0] / 3.,
            )
        except ValueError as error:
            print(error)
            return
    reference_inner_nodes = list(inner_nodes)
    for radial_index in range(patch.radial_layers):
        expected_cap_indices = frozenset((
            cap_nodes[radial_index].index,
            cap_nodes[radial_index + 1].index,
        ))
        if frozenset(cap_edges[radial_index].vertices) != expected_cap_indices:
            print("One-cap edge does not connect consecutive cap nodes")
            return
        if preview_node_rows[radial_index][0] is not cap_nodes[radial_index]:
            print("One-cap preview row does not reuse its cap node")
            return
        if preview_node_rows[radial_index + 1][0] is not cap_nodes[radial_index + 1]:
            print("One-cap preview row does not reuse its next cap node")
            return
        if not np.allclose(
            np_point(preview_node_rows[radial_index + 1][0].position),
            np_point(cap_nodes[radial_index + 1].position),
        ):
            print("One-cap target row does not start at its fixed cap node")
            return

    first_new_element_index = len(element_list)
    node_rows = [inner_nodes]
    for radial_index in range(patch.radial_layers):
        if inner_nodes[0] is not cap_nodes[radial_index]:
            raise ValueError("One-cap row does not start at the expected cap node")
        target_positions = [
            node.position for node in preview_node_rows[radial_index + 1]
        ]
        row_tangents = row_parameters = None
        fixed_cap_effective_vector = None
        if patch.bezier_mode and patch.one_cap_global_active:
            row_tangents, row_parameters, unused_segments = (
                cap_first_global_bezier_row_data(
                    patch, topology, radial_index + 1
                )
            )
            first_interval = row_parameters[1] - row_parameters[0]
            fixed_cap_effective_vector = (
                first_interval * np.asarray(row_tangents[0]) / 3.0
            )
        elif patch.bezier_mode:
            if radial_index == patch.radial_layers - 1:
                row_tangents, row_parameters = (
                    working_tangents, working_parameters
                )
            fixed_cap_effective_vector = (
                (1.0 if topology.start_cap_edges else -1.0)
                * np.asarray(working_vector_rows[radial_index + 1][0])
            )
        row_result = add_one_cap_extended_row(
            inner_nodes, inner_edges, target_positions,
            cap_edges[radial_index], cap_nodes[radial_index + 1],
            reference_inner_nodes=(
                reference_inner_nodes if patch.radial_layers > 1 else None
            ),
            along_tangents=(
                row_tangents
            ),
            along_parameters=(
                row_parameters
            ),
            along_basis_vectors=(
                working_vector_rows[radial_index + 1]
                if working_vector_rows is not None else None
            ),
            fixed_cap_effective_vector=(
                fixed_cap_effective_vector
            ),
        )
        if row_result is None:
            return
        next_nodes, next_edges, unused_created_elements = row_result
        inner_nodes, inner_edges = next_nodes, next_edges
        node_rows.append(inner_nodes)
    created_elements = element_list[first_new_element_index:]

    recompute_node_boundaries(node_list, boundary_list)
    rebuild_node_connections()
    if patch.scene() is scene:
        scene.removeItem(patch)
    view.current_extended_patch = None
    view.selected_edges = []
    return node_rows, created_elements


def _add_single_element_patch_to_nodes_elements(patch):

    corner_count_error = patch_corner_count_error(
        patch.edges, patch.corner_nodes
    )
    if corner_count_error:
        print(corner_count_error)
        return
    if len(patch.edges) == 3:
        ordered_nodes, ordered_edges = ordered_three_edge_chain(patch.edges)
        edge_nodes = [ordered_nodes[3], ordered_nodes[0]]
        edge_vertices = [edge_nodes[0].index, edge_nodes[1].index]
        edge_uv_index = ordered_edges[1].uv_index

        if patch.along_parameter_interval is not None:
            if (
                patch.parameter_start_node is None
                or patch.parameter_end_node is None
            ):
                raise ValueError(
                    "Bezier three-edge closure requires parameter endpoints"
                )
            along_nodal_scales = bezier_nodal_parameter_scales(
                patch.along_parameters
            )
            index = patch.along_edge_index
            edge_sizes = prescribed_bezier_outer_edge_sizes(
                edge_nodes,
                patch.parameter_start_node,
                patch.parameter_end_node,
                patch.along_parameter_interval,
                along_nodal_scales[index],
                along_nodal_scales[index + 1],
            )
            edge_sizes = override_fixed_bezier_endpoint_sizes(
                edge_sizes, edge_nodes,
                patch.parameter_start_node,
                patch.parameter_end_node,
                patch.along_parameter_interval,
                patch.along_tangents[index],
                patch.along_tangents[index + 1],
                edge_uv_index, patch.fixed_bezier_nodes,
                patch.fixed_bezier_effective_vectors,
            )
        else:
            edge_sizes = signed_edge_sizes(edge_nodes, edge_uv_index)
            edge_sizes = override_fixed_endpoint_sizes(
                edge_sizes, edge_nodes, edge_uv_index,
                patch.fixed_bezier_effective_vectors,
            )

        missing_edge = boundary_edge(
            edge_nodes, edge_vertices, [], None, None,
            edge_uv_index, edge_sizes,
        )
        if patch.along_parameter_interval is not None:
            print_bezier_outer_edge_diagnostic(missing_edge, patch)
        scene.addItem(missing_edge)
        patch.edges = ordered_edges + [missing_edge]
        add_element_from_edges(
            patch, ordered_edges, [missing_edge]
        )
        return

    print("number of existing edges : ",len(patch.edges))
    print()

    edges = []
    for this_edge in patch.edges:
        print("edge element_index : ",this_edge.element_index)
        print("edge element_side  : ",this_edge.element_side)   # zero based
        print("edge sizes         : ",this_edge.sizes)
        print("edge uv_index      : ",this_edge.uv_index)
        print("existing edge node 1 :",this_edge.nodes[0].xx[:,0]) 
        print("existing edge node 2 :",this_edge.nodes[1].xx[:,0]) 
        edges.append(this_edge)
        scene.addItem(this_edge)

############################################
# when 2 sides selected : check if they share a node, while not sharing the u or v vector  
# create one new node, two new edges with the new node and the not_shared selected edge nodes.
    if len(edges) == 2:

        inner_edge = patch.edges[0]
        transverse_edge = patch.edges[1]
        shared_node, outer_node0, outer_node1 = two_edge_corner_nodes(
            inner_edge, transverse_edge
        )
        print("two edges forming a corner ", shared_node.index)

        new_node_index = len(node_list)

        direction_0 = outer_node0.xx[:,0] - np_point(patch.corner_nodes[0].position)
        direction_1 = outer_node1.xx[:,0] - np_point(patch.corner_nodes[0].position)

        perp_index_0 = inner_edge.uv_index%2  + 1
        perp_index_1 = transverse_edge.uv_index%2  + 1
        if transverse_edge.uv_index != perp_index_0:
            raise ValueError(
                "Second edge must be transverse to the inner boundary edge"
            )

        print(" direction_0 : ", outer_node0.index, patch.edges[0].uv_index)
        print(" direction_1 : ", outer_node1.index, patch.edges[1].uv_index)

        xx                   = np.zeros((2,4,1))
        xx[:,0,0]            = [patch.corner_nodes[0].position.x(),patch.corner_nodes[0].position.y()]
        xx[:,perp_index_0,0] = direction_0 / 3.
        along_tangent = getattr(patch, "along_tangent", None)
        along_basis_vector = getattr(patch, "along_basis_vector", None)
        along_nodal_scales = (
            bezier_nodal_parameter_scales(patch.along_parameters)
            if along_tangent is not None else None
        )
        along_node_scale = (
            along_nodal_scales[patch.along_edge_index + 1]
            if along_nodal_scales is not None else 1.0
        )
        xx[:,perp_index_1,0] = (
            along_node_scale * np.asarray(along_tangent) / 3.
            if along_tangent is not None
            else np.asarray(along_basis_vector, dtype=float)
            if along_basis_vector is not None
            else direction_1 / 3.
        )
        xx[:,3,0]            = [0.,0.]

        xx = xx / this_scaling
        jorek.nodes_xx = np.append(jorek.nodes_xx,xx,2)
        new_node       = jorek_node_item(new_node_index, this_scaling * xx[:,:,0], 2)
        if (
            (along_tangent is None and along_basis_vector is None)
            or perp_index_0 == 2
        ):
            orient_new_node_red_vector(
                new_node,
                getattr(patch, "radial_reference_node", None),
                (patch.radial_columns[0] if patch.radial_columns else None),
                patch.edges[0].uv_index, perp_index_0,
            )
        node_list.append(new_node)
        scene.addItem(new_node)
        new_node.update()

        edge_vertices = [outer_node0.index, new_node.index]
        edge_nodes    = [outer_node0,       new_node]
        edge_uv_index = perp_index_0

        edge_sizes = np.zeros((2,2))   # order, vertex
        edge_sizes[0,:] = 1.
        edge_sizes[1,0] = inherited_transverse_endpoint_size(
            inner_edge, outer_node0, new_node.position
        )
        edge_sizes[1,1] = np.sign(np.inner(np_point(edge_nodes[0].position - edge_nodes[1].position), edge_nodes[1].xx[:,edge_uv_index]))
        this_edge_2 = boundary_edge(edge_nodes, edge_vertices, [], None, None, edge_uv_index, edge_sizes)
        scene.addItem(this_edge_2)

        edge_vertices = [new_node.index, outer_node1.index]
        edge_nodes    = [new_node,       outer_node1]
        edge_uv_index = perp_index_1

        if patch.along_parameter_interval is not None:
            edge_sizes = prescribed_bezier_outer_edge_sizes(
                edge_nodes, outer_node1, new_node,
                patch.along_parameter_interval,
                along_nodal_scales[patch.along_edge_index],
                along_nodal_scales[patch.along_edge_index + 1],
            )
            edge_sizes = override_fixed_bezier_endpoint_sizes(
                edge_sizes, edge_nodes, outer_node1, new_node,
                patch.along_parameter_interval,
                patch.along_tangents[patch.along_edge_index],
                patch.along_tangents[patch.along_edge_index + 1],
                edge_uv_index, patch.fixed_bezier_nodes,
                patch.fixed_bezier_effective_vectors,
            )
        else:
            edge_sizes = signed_edge_sizes(edge_nodes, edge_uv_index)
            edge_sizes = override_fixed_endpoint_sizes(
                edge_sizes, edge_nodes, edge_uv_index,
                patch.fixed_bezier_effective_vectors,
            )
    
        this_edge_3 = boundary_edge(edge_nodes, edge_vertices, [], None, None, edge_uv_index, edge_sizes)
        if patch.along_parameter_interval is not None:
            print_bezier_outer_edge_diagnostic(this_edge_3, patch)
        scene.addItem(this_edge_3)

        print("exisiting edge 0 ",patch.edges[0].vertices)
        print("exisiting edge 1 ",patch.edges[1].vertices)

        patch.edges.append(this_edge_2)
        patch.edges.append(this_edge_3)

        add_element_from_edges(
            patch, patch.edges[:2], [this_edge_2, this_edge_3]
        )
        return

    elif len(edges) == 1:
        uv_index   = patch.edges[0].uv_index
        perp_index = uv_index%2  + 1

        old_node0 = patch.edges[0].nodes[0]
        old_node1 = patch.edges[0].nodes[1]
        new_at_node0_position, new_at_node1_position = (
            pair_corners_to_edge_endpoints(
                np_point(old_node0.position),
                np_point(old_node1.position),
                np_point(patch.corner_nodes[0].position),
                np_point(patch.corner_nodes[1].position),
            )
        )

        new_at_node1_index = len(node_list)
        direction_to_old_node1 = (
            old_node1.xx[:, 0] - new_at_node1_position
        )
        direction_to_new_at_node0 = (
            new_at_node0_position - new_at_node1_position
        )
        radial_reference_nodes = getattr(
            patch, "radial_reference_nodes", None
        )
        along_tangents = getattr(patch, "along_tangents", None)
        along_basis_vectors = getattr(patch, "along_basis_vectors", None)
        along_nodal_scales = (
            bezier_nodal_parameter_scales(patch.along_parameters)
            if along_tangents is not None else None
        )
        radial_columns = getattr(patch, "radial_columns", None)
        reference_at_node1 = reference_at_node0 = None
        column_at_node1 = column_at_node0 = None
        tangent_at_node1 = tangent_at_node0 = None
        basis_at_node1 = basis_at_node0 = None
        node1_match = node0_match = None
        if (
            radial_reference_nodes is not None
            or along_tangents is not None
            or along_basis_vectors is not None
        ):
            supplied_positions = [
                np_point(node.position) for node in patch.corner_nodes
            ]
            node1_match = next(
                index for index, position in enumerate(supplied_positions)
                if np.allclose(position, new_at_node1_position)
            )
            node0_match = next(
                index for index, position in enumerate(supplied_positions)
                if np.allclose(position, new_at_node0_position)
            )
            if radial_reference_nodes is not None:
                reference_at_node1 = radial_reference_nodes[node1_match]
                reference_at_node0 = radial_reference_nodes[node0_match]
                column_at_node1 = radial_columns[node1_match]
                column_at_node0 = radial_columns[node0_match]
            if along_tangents is not None:
                tangent_at_node1 = along_tangents[node1_match]
                tangent_at_node0 = along_tangents[node0_match]
            if along_basis_vectors is not None:
                basis_at_node1 = along_basis_vectors[node1_match]
                basis_at_node0 = along_basis_vectors[node0_match]
        radial_at_node1 = direction_to_old_node1 / 3.

        xx         = np.zeros((2,4,1))
        xx[:,0,0] = new_at_node1_position
        xx[:,uv_index,0] = (
            along_nodal_scales[
                patch.along_edge_index + node1_match
            ] * np.asarray(tangent_at_node1) / 3.
            if tangent_at_node1 is not None
            else np.asarray(basis_at_node1, dtype=float)
            if basis_at_node1 is not None
            else direction_to_new_at_node0 / 3.
        )
        xx[:,perp_index,0] = radial_at_node1
        xx[:,3,0] = [0.,0.]

        xx = xx / this_scaling
        jorek.nodes_xx = np.append(jorek.nodes_xx,xx,2)
        new_at_node1 = jorek_node_item(
            new_at_node1_index, this_scaling * xx[:,:,0], 2
        )
        if (
            uv_index != 2
            or (tangent_at_node1 is None and basis_at_node1 is None)
        ):
            orient_new_node_red_vector(
                new_at_node1, reference_at_node1, column_at_node1,
                uv_index, perp_index,
            )
        node_list.append(new_at_node1)
        scene.addItem(new_at_node1)
        new_at_node1.update()
    
        new_at_node0_index = len(node_list)
        direction_to_old_node0 = (
            old_node0.xx[:, 0] - new_at_node0_position
        )
        direction_to_new_at_node1 = (
            new_at_node1_position - new_at_node0_position
        )
        radial_at_node0 = direction_to_old_node0 / 3.

        xx         = np.zeros((2,4,1))
        xx[:,0,0] = new_at_node0_position
        xx[:,uv_index,0] = (
            along_nodal_scales[
                patch.along_edge_index + node0_match
            ] * np.asarray(tangent_at_node0) / 3.
            if tangent_at_node0 is not None
            else np.asarray(basis_at_node0, dtype=float)
            if basis_at_node0 is not None
            else direction_to_new_at_node1 / 3.
        )
        xx[:,perp_index,0] = radial_at_node0
        xx[:,3,0] = [0.,0.]

        xx = xx / this_scaling
        jorek.nodes_xx = np.append(jorek.nodes_xx,xx,2)
        new_at_node0 = jorek_node_item(
            new_at_node0_index, this_scaling * xx[:,:,0], 2
        )
        if (
            uv_index != 2
            or (tangent_at_node0 is None and basis_at_node0 is None)
        ):
            orient_new_node_red_vector(
                new_at_node0, reference_at_node0, column_at_node0,
                uv_index, perp_index,
            )
        node_list.append(new_at_node0)
        scene.addItem(new_at_node0)
        new_at_node0.update()

# new_at_node1 -> new_at_node0
        edge_vertices = [new_at_node1.index, new_at_node0.index]
        edge_nodes    = [new_at_node1,       new_at_node0]
        edge_uv_index = uv_index

        if patch.along_parameter_interval is not None:
            parameter_start_node = (
                new_at_node1 if node1_match == 0 else new_at_node0
            )
            parameter_end_node = (
                new_at_node1 if node1_match == 1 else new_at_node0
            )
            edge_sizes = prescribed_bezier_outer_edge_sizes(
                edge_nodes, parameter_start_node, parameter_end_node,
                patch.along_parameter_interval,
                along_nodal_scales[patch.along_edge_index],
                along_nodal_scales[patch.along_edge_index + 1],
            )
        else:
            edge_sizes = signed_edge_sizes(edge_nodes, edge_uv_index)
    
        this_edge_3 = boundary_edge(edge_nodes, edge_vertices, [], None, None, edge_uv_index, edge_sizes)
        if patch.along_parameter_interval is not None:
            print_bezier_outer_edge_diagnostic(this_edge_3, patch)
        scene.addItem(this_edge_3)

# old_node1 -> new_at_node1
        edge_vertices = [old_node1.index, new_at_node1.index]
        edge_nodes    = [old_node1,       new_at_node1]
        edge_uv_index = perp_index

        edge_sizes = np.zeros((2,2))   # order, vertex
        edge_sizes[0,:] = 1.
        edge_sizes[1,0] = inherited_transverse_endpoint_size(
            patch.edges[0], old_node1, new_at_node1.position
        )
        edge_sizes[1,1] = np.sign(np.inner(np_point(edge_nodes[0].position - edge_nodes[1].position), edge_nodes[1].xx[:,edge_uv_index]))
    
        this_edge_2 = boundary_edge(edge_nodes, edge_vertices, [], None, None, perp_index, edge_sizes)
        scene.addItem(this_edge_2)

# new_at_node0 -> old_node0
        edge_vertices = [new_at_node0.index, old_node0.index]
        edge_nodes    = [new_at_node0,       old_node0]
        edge_uv_index = perp_index

        edge_sizes = np.zeros((2,2))   # order, vertex
        edge_sizes[0,:] = 1.
        edge_sizes[1,0] = np.sign(np.inner(np_point(edge_nodes[1].position - edge_nodes[0].position), edge_nodes[0].xx[:,edge_uv_index]))
        edge_sizes[1,1] = inherited_transverse_endpoint_size(
            patch.edges[0], old_node0, new_at_node0.position
        )
    
        this_edge_4 = boundary_edge(edge_nodes, edge_vertices, [], None, None, perp_index, edge_sizes)
        scene.addItem(this_edge_4)
 
        patch.edges.append(this_edge_2)
        patch.edges.append(this_edge_3)
        patch.edges.append(this_edge_4)

        add_element_from_edges(
            patch, [patch.edges[0]], [this_edge_2, this_edge_3, this_edge_4]
        )


def assign_element_edge_metadata(edges, order, element_index, vertices):
    for element_side, edge_index in enumerate(order):
        edge = edges[edge_index]
        local_nodes_index = [element_side, (element_side + 1) % 4]
        expected_vertices = [
            vertices[local_nodes_index[0]], vertices[local_nodes_index[1]]
        ]
        if list(edge.vertices) != expected_vertices:
            raise ValueError("ordered edge vertices do not match element side")
        edge.local_nodes_index = local_nodes_index
        edge.element_index = element_index
        edge.element_side = element_side


def add_element_from_edges(patch, existing_edges, new_edges):

    patch.edges = order_edges(patch.edges)

    element_index = max(
        (element.index for element in element_list), default=-1
    ) + 1

    order = [0,1,2,3]
    if patch.edges[0].uv_index == 2: order = [3,0,1,2]

    nodes    = [patch.edges[order[0]].nodes[0], patch.edges[order[1]].nodes[0], 
                patch.edges[order[2]].nodes[0], patch.edges[order[3]].nodes[0]] 
    vertices = [nodes[0].index, nodes[1].index, nodes[2].index, nodes[3].index] 

    assign_element_edge_metadata(
        patch.edges, order, element_index, vertices
    )

    print("vertices       : ",vertices)
    for this_node in nodes:
        print("nodes.boundary : ",this_node.index, this_node.boundary)

    sizes         = np.zeros((4,4))
    sizes[0,:]    = 1.    #   sizes(order, vertex)
  
    sizes[1,0] = patch.edges[order[0]].sizes[1,0]
    sizes[1,1] = patch.edges[order[0]].sizes[1,1]  

    sizes[2,1] = patch.edges[order[1]].sizes[1,0] 
    sizes[2,2] = patch.edges[order[1]].sizes[1,1]    

    sizes[1,2] = patch.edges[order[2]].sizes[1,0]    
    sizes[1,3] = patch.edges[order[2]].sizes[1,1] 

    sizes[2,3] = patch.edges[order[3]].sizes[1,0]  
    sizes[2,0] = patch.edges[order[3]].sizes[1,1]

    sizes[3,0] = sizes[1,0] * sizes[2,0]  
    sizes[3,2] = sizes[1,2] * sizes[2,2]  
    sizes[3,1] = sizes[1,1] * sizes[2,1]  
    sizes[3,3] = sizes[1,3] * sizes[2,3]  

    new_element   = jorek_element_item(element_index, vertices, sizes)
    scene.addItem(new_element)
    element_list.append(new_element)
    new_element.update()

    for edge in existing_edges:
        print("removing edge ", edge.vertices)
        deactivate_boundary_edge(edge)
        boundary_list.remove(edge)

    for edge in new_edges:
        print(" add new edges : ", edge.vertices)
        boundary_list.append(edge)
        new_element.edges.append(edge)

    recompute_node_boundaries(node_list, boundary_list)

    rebuild_node_connections()

    if (
        view.current_patch is not None
        and isinstance(view.current_patch, QGraphicsItem)
    ):
        scene.removeItem(view.current_patch)
    view.current_patch  = None
    view.selected_edges = []

    return


def recompute_node_boundaries(nodes, boundary_edges):
    """Rebuild scalar node boundary flags from the final boundary-edge set."""
    directions_by_node = {node.index: set() for node in nodes}
    for node in nodes:
        node.boundary = 0

    for edge in boundary_edges:
        if not getattr(edge, "active", True):
            continue
        for node in edge.nodes:
            directions_by_node[node.index].add(edge.uv_index)

    for node in nodes:
        directions = directions_by_node[node.index]
        if len(directions) == 1:
            node.boundary = next(iter(directions))
        elif len(directions) > 1:
            # JOREK stores one scalar boundary marker.  Existing grids and
            # newly created corner nodes use 2 when both directions meet.
            node.boundary = 2


def element_side_key(element, element_side):
    return frozenset((
        element.vertices[element_side],
        element.vertices[(element_side + 1) % 4],
    ))


def boundary_edge_from_element_side(element, element_side):
    local_node0 = element_side
    local_node1 = (element_side + 1) % 4
    vertices = [
        element.vertices[local_node0], element.vertices[local_node1]
    ]
    nodes = [node_list[vertices[0]], node_list[vertices[1]]]
    uv_index = element_side % 2 + 1
    edge_sizes = np.zeros((2,2))
    edge_sizes[:,0] = [
        element.sizes[0,local_node0], element.sizes[uv_index,local_node0]
    ]
    edge_sizes[:,1] = [
        element.sizes[0,local_node1], element.sizes[uv_index,local_node1]
    ]
    return boundary_edge(
        nodes, vertices, [local_node0, local_node1],
        element.index, element_side, uv_index, edge_sizes,
    )


def deactivate_node(node):
    """Keep an unused node scene-owned but make all of its graphics inactive."""
    active_view = globals().get("view")
    if active_view is not None:
        if getattr(active_view, "dragged_node", None) is node:
            active_view.dragged_node = None
        active_view.selected_nodes = [
            selected
            for selected in (getattr(active_view, "selected_nodes", None) or [])
            if selected is not node
        ]
        selected_point = getattr(active_view, "selected_point", None)
        if any(selected_point is item for item in (
            node,
            getattr(node, "blue_handle", None),
            getattr(node, "red_handle", None),
        )):
            active_view.selected_point = None
    node.connected_elements = []
    node.connected_boundary_edges = []
    node.active = False
    if isinstance(node, QGraphicsItem):
        node.setVisible(False)
        node.setEnabled(False)


def deactivate_element(element):
    """Remove an element from interaction while preserving its Qt object."""
    active_view = globals().get("view")
    if active_view is not None:
        active_view.selected_elements = [
            selected
            for selected in (
                getattr(active_view, "selected_elements", None) or []
            )
            if selected is not element
        ]
    element.active = False
    if isinstance(element, QGraphicsItem):
        element.setVisible(False)
        element.setEnabled(False)
        element.path_item.setVisible(False)


def deactivate_boundary_edge(edge):
    """Remove a boundary edge from interaction while preserving its Qt object."""
    active_view = globals().get("view")
    if active_view is not None:
        active_view.selected_edges = [
            selected
            for selected in (getattr(active_view, "selected_edges", None) or [])
            if selected is not edge
        ]
    edge.active = False
    edge.setVisible(False)
    edge.setEnabled(False)


def clear_deleted_item_selection_references(this_view, items, removed_nodes):
    """Drop view references to graphics items before removing them from Qt."""
    items = list(items)

    def retained(item):
        return not any(item is removed_item for removed_item in items)

    this_view.selected_nodes = [
        node for node in (getattr(this_view, "selected_nodes", None) or [])
        if retained(node)
    ]
    this_view.selected_elements = [
        element
        for element in (getattr(this_view, "selected_elements", None) or [])
        if retained(element)
    ]
    this_view.selected_edges = [
        edge for edge in (getattr(this_view, "selected_edges", None) or [])
        if retained(edge)
    ]

    selected_point = getattr(this_view, "selected_point", None)
    removed_node_graphics = [
        graphic
        for node in removed_nodes
        for graphic in (
            node,
            getattr(node, "blue_handle", None),
            getattr(node, "red_handle", None),
            getattr(node, "ellipse_item", None),
        )
        if graphic is not None
    ]
    if selected_point is not None and any(
        selected_point is item for item in items + removed_node_graphics
    ):
        this_view.selected_point = None


def delete_selected_element(this_view):
    selected_elements = this_view.selected_elements
    if selected_elements is None or len(selected_elements) != 1:
        print("Select exactly one element to delete")
        return

    deleted_element = selected_elements[0]
    if deleted_element not in element_list:
        print("The selected element is no longer present")
        return

    remaining_elements = [
        element for element in element_list if element is not deleted_element
    ]
    referenced_nodes = {
        vertex for element in remaining_elements for vertex in element.vertices
    }
    removed_nodes = [
        node for node in node_list
        if (
            node.index not in referenced_nodes
            and getattr(node, "active", True)
        )
    ]
    exposed_sides = []
    deleted_side_keys = {
        element_side_key(deleted_element, side) for side in range(4)
    }
    for side_key in deleted_side_keys:
        neighbours = [
            (element, side)
            for element in remaining_elements
            for side in range(4)
            if element_side_key(element, side) == side_key
        ]
        if len(neighbours) > 1:
            print("Cannot delete an element from a non-manifold grid")
            return
        if neighbours:
            exposed_sides.append(neighbours[0])

    removed_edges = [
        edge for edge in boundary_list
        if frozenset(edge.vertices) in deleted_side_keys
    ]
    clear_deleted_item_selection_references(
        this_view, removed_edges + [deleted_element] + removed_nodes,
        removed_nodes,
    )

    for edge in removed_edges:
        print(
            "before edge.setVisible(False) {}".format(edge.vertices),
            flush=True,
        )
        deactivate_boundary_edge(edge)
        print(
            "after edge.setVisible(False) {}".format(edge.vertices),
            flush=True,
        )
        boundary_list.remove(edge)

    element_list.remove(deleted_element)
    print(
        "before element.setVisible(False) {}".format(deleted_element.index),
        flush=True,
    )
    deactivate_element(deleted_element)
    print(
        "after element.setVisible(False) {}".format(deleted_element.index),
        flush=True,
    )

    for element, side in exposed_sides:
        print(
            "before boundary_edge_from_element_side({}, {})".format(
                element.index, side
            ),
            flush=True,
        )
        new_edge = boundary_edge_from_element_side(element, side)
        print(
            "after boundary_edge_from_element_side({}, {})".format(
                element.index, side
            ),
            flush=True,
        )
        boundary_list.append(new_edge)
        element.edges.append(new_edge)
        print(
            "before scene.addItem(new edge {})".format(new_edge.vertices),
            flush=True,
        )
        scene.addItem(new_edge)
        print(
            "after scene.addItem(new edge {})".format(new_edge.vertices),
            flush=True,
        )

    for node in removed_nodes:
        deactivate_node(node)

    print("before recompute_node_boundaries", flush=True)
    recompute_node_boundaries(node_list, boundary_list)
    print("after recompute_node_boundaries", flush=True)
    print("before rebuild_graphics_layers", flush=True)
    rebuild_graphics_layers(
        active_view=this_view, topology_changed=True
    )
    print("after rebuild_graphics_layers", flush=True)
    this_view.mark_document_modified()

def order_edges(edges):
    ordered_edges = []
    next_vertex = edges[0].vertices[1]
    ordered_edges.append(edges[0])
    edges.remove(edges[0])
 
    while edges:
        for edge in edges:
            if (edge.vertices[0] == next_vertex):
                next_vertex = edge.vertices[1]
                ordered_edges.append(edge)
                edges.remove(edge)
            elif (edge.vertices[1] == next_vertex):
                edge = reversed_edge(edge)
                next_vertex = edge.vertices[1]
                ordered_edges.append(edge)
                edges.remove(edge)
    return ordered_edges

def distance(position1, position2):
    return np.sqrt((position1.x() - position2.x())**2 + (position1.y() - position2.y())**2 )

def np_point(qpointf):
    return np.array([qpointf.x(), qpointf.y()])
    
def qt_point(x):
    return QPointF(x[0].item(), x[1].item())


class mesh_node_record:
    """Non-graphics node record preserving node_list[index] semantics."""
    def __init__(self, index, xx, boundary, active=True):
        self.index = index
        self.xx = np.asarray(xx)
        self.position = QPointF(self.xx[0, 0], self.xx[1, 0])
        self.boundary = boundary
        self.active = active
        self.connected_elements = []
        self.connected_boundary_edges = []

    def prepareGeometryChange(self):
        pass

    def update(self):
        pass


class mesh_element_record:
    """Non-graphics topology record for a frozen interior element."""
    def __init__(self, index, vertices, sizes, active=True):
        self.index = index
        self.vertices = np.asarray(vertices)
        self.sizes = np.asarray(sizes)
        self.active = active
        self.edges = []

    def update(self):
        pass

    def scene(self):
        return None


def element_edge_adjacency(elements):
    """Return active-element neighbours connected by complete quad edges."""
    active_elements = [
        element for element in elements if getattr(element, "active", True)
    ]
    adjacency = {element.index: set() for element in active_elements}
    owners_by_edge = {}
    for element in active_elements:
        if len(element.vertices) != 4:
            raise ValueError("Editable adjacency requires quadrilateral elements")
        for side in range(4):
            edge_key = frozenset((
                int(element.vertices[side]),
                int(element.vertices[(side + 1) % 4]),
            ))
            owners_by_edge.setdefault(edge_key, []).append(element.index)

    for owners in owners_by_edge.values():
        for owner_index, owner in enumerate(owners):
            adjacency[owner].update(owners[:owner_index])
            adjacency[owner].update(owners[owner_index + 1:])
    return adjacency


def expand_editable_element_indices(starting_indices, adjacency, depth):
    depth = int(depth)
    if depth < 0:
        raise ValueError("Editable depth must be nonnegative")
    editable = set(starting_indices).intersection(adjacency)
    frontier = set(editable)
    for unused_layer in range(depth):
        next_frontier = {
            neighbour
            for element_index in frontier
            for neighbour in adjacency.get(element_index, ())
            if neighbour not in editable
        }
        if not next_frontier:
            break
        editable.update(next_frontier)
        frontier = next_frontier
    return editable


def editable_element_indices(
    elements, boundary_edges, depth=0, adjacency=None,
):
    """Return active boundary owners plus ``depth`` shared-edge rings."""
    active_indices = {
        element.index for element in elements
        if getattr(element, "active", True)
    }
    boundary_owners = {
        edge.element_index for edge in boundary_edges
        if (
            getattr(edge, "active", True)
            and edge.element_index in active_indices
        )
    }
    if adjacency is None:
        adjacency = element_edge_adjacency(elements)
    return expand_editable_element_indices(
        boundary_owners, adjacency, depth
    )


def editable_boundary_element_indices(elements, boundary_edges):
    """Backward-compatible name for the depth-0 editable element set."""
    return editable_element_indices(elements, boundary_edges, depth=0)


def editable_node_indices(elements, editable_element_indices):
    """Union of vertices belonging to the editable element set."""
    editable_element_indices = set(editable_element_indices)
    return {
        int(vertex)
        for element in elements
        if (
            getattr(element, "active", True)
            and element.index in editable_element_indices
        )
        for vertex in element.vertices
    }

def reversed_edge(edge):
    edge.vertices          = edge.vertices[::-1]
    edge.nodes             = edge.nodes[::-1] 
    edge.local_nodes_index = edge.local_nodes_index[::-1]
    edge.sizes[0,:]        = edge.sizes[0,::-1]
    edge.sizes[1,:]        = edge.sizes[1,::-1]

    edge.points = jorek.nodes_xx[:,0:edge.uv_index+1:edge.uv_index,edge.vertices] * this_scaling
    edge.points[0,:,:] = edge.points[0,:,:] * edge.sizes  
    edge.points[1,:,:] = edge.points[1,:,:] * edge.sizes
    return edge


def node_display_reference(node):
    """Return a deterministic active (element, local vertex) display owner.

    Boundary-edge owners are preferred for boundary nodes.  Within either
    candidate set, the element with the lowest persistent index wins.
    """
    candidates = []
    for element in getattr(node, "connected_elements", []):
        if not getattr(element, "active", True):
            continue
        local_vertices = [
            local_vertex
            for local_vertex, vertex in enumerate(element.vertices)
            if vertex == node.index
        ]
        if local_vertices:
            candidates.append((element, local_vertices[0]))
    if not candidates:
        return None

    if node.boundary:
        boundary_owner_indices = {
            edge.element_index
            for edge in getattr(node, "connected_boundary_edges", [])
            if (
                getattr(edge, "active", True)
                and edge.element_index is not None
                and node.index in edge.vertices
            )
        }
        boundary_candidates = [
            candidate for candidate in candidates
            if candidate[0].index in boundary_owner_indices
        ]
        if boundary_candidates:
            candidates = boundary_candidates

    return min(
        candidates,
        key=lambda candidate: (candidate[0].index, candidate[1]),
    )


def node_basis_display_scale(node, basis_index):
    """Return the owning element's signed nodal scale for one basis."""
    if basis_index not in (1, 2):
        raise ValueError("A displayed nodal basis index must be 1 or 2")
    reference = node_display_reference(node)
    if reference is None:
        # Orphan/new nodes have no element-local parameter scale yet.
        return 1.0
    element, local_vertex = reference
    return float(element.sizes[basis_index, local_vertex])


def node_basis_display_vector(node, basis_index):
    return (
        node_basis_display_scale(node, basis_index)
        * node.xx[:, basis_index]
    )


def rebuild_node_connections():
    """Cache the drawable items affected by changes to each node."""
    for node in node_list:
        node.prepareGeometryChange()
        node.connected_elements = []
        node.connected_boundary_edges = []

    for element in element_list:
        if not getattr(element, "active", True):
            continue
        for vertex in element.vertices:
            node_list[vertex].connected_elements.append(element)

    for edge in boundary_list:
        if not getattr(edge, "active", True):
            continue
        for vertex in edge.vertices:
            node_list[vertex].connected_boundary_edges.append(edge)

    for node in node_list:
        if not getattr(node, "active", True):
            continue
        if isinstance(node, jorek_node_item):
            node.blue_handle.sync_position()
            node.red_handle.sync_position()
        node.update()

class boundary_edge(QGraphicsPathItem):
    def __init__(self, nodes, vertices, local_nodes_index, element_index, element_side, uv_index, element_sizes):
        super().__init__()
        self.setZValue(Z_BOUNDARY_EDGE)
        self.active = True
        self.vertices          = vertices
        self.nodes             = nodes 
        self.local_nodes_index = local_nodes_index
        self.element_index     = element_index
        self.element_side      = element_side
        self.uv_index          = uv_index
        self.sizes             = element_sizes
        
        self.points = jorek.nodes_xx[:,0:uv_index+1:uv_index,vertices] * this_scaling

        self.points[0,:,:] = self.points[0,:,:] * self.sizes   # [x, order, vertex]
        self.points[1,:,:] = self.points[1,:,:] * self.sizes

        self.setFlag(QGraphicsItem.ItemIsMovable,False)
        self.setFlag(QGraphicsItem.ItemIsSelectable,True)
        self.setPen(boundary_edge_pen())
        if DIAGNOSTIC_ELEMENTS_ONLY:
            self.setVisible(False)

        self.setPath(self.createPath())

    def createPath(self):
        path = QPainterPath()

        bezier_points = edge_bezier_points(self.points)

        path.moveTo( QPointF(bezier_points[0,0,0].item(),bezier_points[1,0,0].item()))
        path.cubicTo(QPointF(bezier_points[0,1,0].item(),bezier_points[1,1,0].item()),
                     QPointF(bezier_points[0,1,1].item(),bezier_points[1,1,1].item()),
                     QPointF(bezier_points[0,0,1].item(),bezier_points[1,0,1].item()))
        return path

    def shape(self):
        pen = self.pen()
        stroke_width = pen.widthF()
        if pen.isCosmetic():
            zoom_level = 1.0
            if self.scene() and self.scene().views():
                zoom_level = self.scene().views()[0].zoom_level
            stroke_width /= zoom_level
        stroker = QPainterPathStroker()
        stroker.setWidth(stroke_width)
        stroker.setCapStyle(pen.capStyle())
        stroker.setJoinStyle(pen.joinStyle())
        stroker.setMiterLimit(pen.miterLimit())
        return stroker.createStroke(self.path())

    def paint(self, painter: QPainter, option, widget=None):        
        self.points = jorek.nodes_xx[:,0:self.uv_index+1:self.uv_index,self.vertices] * this_scaling
        self.points[0,:,:] = self.points[0,:,:] * self.sizes   # [x, order, vertex]
        self.points[1,:,:] = self.points[1,:,:] * self.sizes
        self.setPath(self.createPath())

        painter.setPen(self.pen())
        painter.drawPath(self.path())

        if SHOW_EDGE_INDICES:
            mid_edge = np.sum(self.points[:,0,:],1) / 2.
            painter.setPen(QPen(Qt.green))
            painter.drawText(
                qt_point(0.7 * self.points[:,0,0] + 0.3 * mid_edge), "1"
            )
            painter.drawText(
                qt_point(0.7 * self.points[:,0,1] + 0.3 * mid_edge), "2"
            )


def scaled_element_points(vertices, sizes):
    points = jorek.nodes_xx[:, :, vertices]
    points[0, :, :] *= sizes
    points[1, :, :] *= sizes
    return points


def element_outline_path(points):
    bezier_points = element_bezier_points(points, this_scaling)
    path = QPainterPath()
    path.moveTo(qt_point(bezier_points[:, 0, 0]))
    path.cubicTo(
        qt_point(bezier_points[:, 1, 0]),
        qt_point(bezier_points[:, 2, 0]),
        qt_point(bezier_points[:, 3, 0]),
    )
    path.cubicTo(
        qt_point(bezier_points[:, 3, 1]),
        qt_point(bezier_points[:, 3, 2]),
        qt_point(bezier_points[:, 3, 3]),
    )
    path.cubicTo(
        qt_point(bezier_points[:, 2, 3]),
        qt_point(bezier_points[:, 1, 3]),
        qt_point(bezier_points[:, 0, 3]),
    )
    path.cubicTo(
        qt_point(bezier_points[:, 0, 2]),
        qt_point(bezier_points[:, 0, 1]),
        qt_point(bezier_points[:, 0, 0]),
    )
    return path


class static_mesh_item(QGraphicsPathItem):
    """Single non-interactive path containing every active mesh element."""
    def __init__(self):
        super().__init__()
        self.setZValue(Z_STATIC_MESH)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        pen = QPen(QColor(70, 70, 70))
        pen.setWidthF(STATIC_MESH_WIDTH)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.rebuild_path()

    def rebuild_path(self):
        path = QPainterPath()
        for element in element_list:
            if not getattr(element, "active", True):
                continue
            path.addPath(element_outline_path(
                scaled_element_points(element.vertices, element.sizes)
            ))
        self.setPath(path)


def rebuild_static_mesh_path(expected_scene=None):
    current_static_mesh = globals().get("static_mesh")
    if (
        current_static_mesh is not None
        and (
            expected_scene is None
            or graphics_item_in_scene(current_static_mesh, expected_scene)
        )
    ):
        current_static_mesh.rebuild_path()


class jorek_element_item(QGraphicsItem):
    def __init__(self, index, vertices, sizes):
        super().__init__()
        self.setZValue(Z_MESH_EDGE)
        self.active = True

        self.index    = index
        self.vertices = vertices      # [vertex,order]
        self.sizes    = sizes
        self.points   = jorek.nodes_xx[:,:,vertices]
        self.edges    = []

#        self.points = np.array([node_list[vertices[0]].xx,
#                                node_list[vertices[1]].xx,
#                                node_list[vertices[2]].xx,
#                                node_list[vertices[3]].xx])  / this_scaling 
#        self.points = np.transpose(np.swapaxes(self.points,1,2))

        self.points[0,:,:] = self.points[0,:,:] * self.sizes   # [x,order,vertex]
        self.points[1,:,:] = self.points[1,:,:] * self.sizes
        
        self.path_item = QGraphicsPathItem()
        self.setFlag(QGraphicsItem.ItemIsMovable,False)
        self.setFlag(QGraphicsItem.ItemIsSelectable,False)
        self.path_item.setBrush(QBrush(QColor(255, 255, 255, 64)))

        self.setAcceptHoverEvents(False)
        self.setEnabled(False)

        self.path_item.setPath(self.createPath())

    
    def find_boundary_edges(self):
        boundary_edges = []
        for i_side in range(4):
            local_node_1 =  i_side
            local_node_2 = (i_side+1)%4
            node_1 = self.vertices[local_node_1]
            node_2 = self.vertices[local_node_2]
            nodes = [node_list[node_1], node_list[node_2]]
            edge_sizes = np.zeros((2,2))   # order, vertex
            if node_list[node_1].boundary and node_list[node_2].boundary:
                uv_index = i_side%2 + 1
                edge_sizes[:,0] = [self.sizes[0,local_node_1],self.sizes[uv_index,local_node_1]]
                edge_sizes[:,1] = [self.sizes[0,local_node_1],self.sizes[uv_index,local_node_2]]
                this_edge = boundary_edge(nodes, [node_1,node_2], [i_side,(i_side+1)%4], self.index, i_side, uv_index, edge_sizes)
                boundary_edges.append(this_edge)   
#                print("boundary edge : ",nodes,this_edge.element_index, this_edge.element_side, this_edge.vertices,this_edge.local_nodes_index, edge_sizes)
        return boundary_edges 

  
    def createPath(self):
        return element_outline_path(self.points)
    
    def boundingRect(self):
        return self.path_item.boundingRect()

    def paint(self, painter: QPainter, option, widget=None):        
        self.points        = jorek.nodes_xx[:,:,self.vertices]
        self.points[0,:,:] = self.points[0,:,:] * self.sizes   # [x,order,vertex]
        self.points[1,:,:] = self.points[1,:,:] * self.sizes
        self.path_item.setPath(self.createPath())
 
        zoom_level = self.scene().views()[0].zoom_level 
        self.path_item.setPen(QPen(Qt.black, 1./zoom_level))

        painter.setPen(self.path_item.pen())
        painter.setBrush(self.path_item.brush())
        painter.drawPath(self.path_item.path())

        center = np.sum(self.points[:,0,:],1) / 4.
    #    painter.drawText(qt_point(this_scaling*center),str(self.index))
    #    for i in range(4):
    #        painter.drawText(qt_point(this_scaling*(0.8*self.points[:,0,i]+0.2*center)),str(i+1))        

        painter.setPen(QPen(Qt.red))
        for i in range(4):
            mid_edge = (self.points[:,0,i]+self.points[:,0,(i+1)%4]) / 2.
    #        painter.drawText(qt_point(this_scaling*(0.8*mid_edge+0.2*center)),str(i+1))        

    def mousePressEvent(self, event: QMouseEvent) -> None:
        print('element : mousepressevent')
        self.mouse_pos = event.pos()
#        event.accept()


class basis_vector_handle(QGraphicsEllipseItem):
    def __init__(self, node, basis_index, color):
        super().__init__(
            -VECTOR_HANDLE_SIZE / 2.0, -VECTOR_HANDLE_SIZE / 2.0,
            VECTOR_HANDLE_SIZE, VECTOR_HANDLE_SIZE, node,
        )
        self.node = node
        self.basis_index = basis_index
        self.setBrush(QBrush(color))
        self.setPen(graphics_handle_outline_pen())
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(Z_VECTOR_HANDLE)
        self.sync_position()

    def sync_position(self):
        effective = node_basis_display_vector(self.node, self.basis_index)
        endpoint = self.node.position + qt_point(effective)
        self.setPos(endpoint)

    def move_to_scene(self, scene_position):
        if (
            not getattr(self.node, "active", True)
            or self.scene() is None
            or self.node.scene() is None
        ):
            return
        node_position = np_point(self.node.position)
        handle_position = np_point(scene_position)
        vector = basis_vector_from_scene(
            node_position, handle_position, this_scaling
        )
        reference_scale = node_basis_display_scale(
            self.node, self.basis_index
        )
        if np.isclose(reference_scale, 0.0):
            print(
                "Cannot move basis handle with zero reference scale:",
                self.node.index, self.basis_index,
            )
            self.sync_position()
            return
        raw_vector = vector / reference_scale

        self.node.prepareGeometryChange()
        jorek.nodes_xx[:, self.basis_index, self.node.index] = raw_vector
        self.node.xx[:, self.basis_index] = this_scaling * raw_vector
        self.setPos(scene_position)
        self.node.update_connected_items(self.basis_index)
        scene = self.node.scene()
        if scene is not None and scene.views():
            scene.views()[0].mark_document_modified()


class jorek_node_item(QGraphicsItem):
    def __init__(self, index, xx, boundary):
        super().__init__()
        self.setZValue(Z_NODE)

        self.ellipse_size  = 30
        self.xx           = xx
        self.boundary     = boundary
        self.connected_elements = []
        self.connected_boundary_edges = []
        self.active = True
        self.ellipse_item = QGraphicsEllipseItem(xx[0,0] - self.ellipse_size/2, xx[1,0] - self.ellipse_size/2, self.ellipse_size, self.ellipse_size)
        self.ellipse_item.setFlag(QGraphicsItem.ItemIsMovable)
        self.ellipse_item.setFlag(QGraphicsItem.ItemIsSelectable)

        self.ellipse_item.setPen(QPen(Qt.black, 1.))
        self.ellipse_item.setBrush(QBrush(NODE_COLOR))

        self.index = index

#        self.path_item.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
#        self.path_item.setAcceptHoverEvents(True)

        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.position = QPointF(xx[0,0], xx[1,0])
        self.blue_handle = basis_vector_handle(self, 1, QColor(0, 0, 255))
        self.red_handle = basis_vector_handle(self, 2, QColor(255, 0, 0))
        if DIAGNOSTIC_ELEMENTS_ONLY:
            self.setVisible(False)

    def boundingRect(self):
        zoom_level = 1.0
        if self.scene() and self.scene().views():
            zoom_level = self.scene().views()[0].zoom_level

        node_radius = NODE_MARKER_SIZE / (2.0 * zoom_level)
        handle_radius = VECTOR_HANDLE_SIZE / (2.0 * zoom_level)
        pen_margin = 6.0 / zoom_level
        blue_endpoint = self.position + qt_point(
            node_basis_display_vector(self, 1)
        )
        red_endpoint = self.position + qt_point(
            node_basis_display_vector(self, 2)
        )

        left = min(
            self.position.x() - node_radius,
            blue_endpoint.x() - handle_radius,
            red_endpoint.x() - handle_radius,
        )
        top = min(
            self.position.y() - node_radius,
            blue_endpoint.y() - handle_radius,
            red_endpoint.y() - handle_radius,
        )
        right = max(
            self.position.x() + node_radius,
            blue_endpoint.x() + handle_radius,
            red_endpoint.x() + handle_radius,
        )
        bottom = max(
            self.position.y() + node_radius,
            blue_endpoint.y() + handle_radius,
            red_endpoint.y() + handle_radius,
        )
        return QRectF(
            left - pen_margin,
            top - pen_margin,
            right - left + 2.0 * pen_margin,
            bottom - top + 2.0 * pen_margin,
        )

    def shape(self):
        zoom_level = 1.0
        if self.scene() and self.scene().views():
            zoom_level = self.scene().views()[0].zoom_level

        radius = NODE_MARKER_SIZE / (2.0 * zoom_level)
        path = QPainterPath()
        path.addEllipse(self.position, radius, radius)
        return path
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not getattr(self, "active", True):
            return
        print('node : mousePressEvent')
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not getattr(self, "active", True):
            return
        scene = self.scene()
        if scene is None or not scene.views():
            return
        self.move_to_scene(scene.views()[0].mapToScene(event.pos()))
        event.accept()

    def move_to_scene(self, scene_position):
        if not getattr(self, "active", True):
            return
        self.prepareGeometryChange()
        self.position = scene_position
        self.ellipse_item.setPos(self.position.x(), self.position.y())
        jorek.nodes_xx[0,0,self.index] = self.position.x() / this_scaling
        jorek.nodes_xx[1,0,self.index] = self.position.y() / this_scaling
        self.blue_handle.sync_position()
        self.red_handle.sync_position()
        self.update_connected_items()
        scene = self.scene()
        if scene is not None and scene.views():
            scene.views()[0].mark_document_modified()

    def update_connected_items(self, basis_index=None):
        if self.scene() is not None:
            self.update()
        for element in self.connected_elements:
            if not hasattr(element, "scene") or element.scene() is not None:
                element.update()
        for edge in self.connected_boundary_edges:
            if (
                (not hasattr(edge, "scene") or edge.scene() is not None)
                and (basis_index is None or edge.uv_index == basis_index)
            ):
                edge.update()
    

    def paint(self, painter: QPainter, option, widget=None):

        scene = self.scene()
        if scene is None or not scene.views():
            return

   #     self.prepareGeometryChange()

        zoom_level = scene.views()[0].zoom_level 
        marker_size = NODE_MARKER_SIZE / zoom_level
        self.ellipse_size = marker_size
        self.ellipse_item.setRect(self.position.x()-self.ellipse_size/2, self.position.y()-self.ellipse_size/2,
                                  self.ellipse_size, self.ellipse_size)

        painter.setPen(QPen(Qt.black, 1. / zoom_level))

        painter.setPen(QPen(Qt.blue, BASIS_VECTOR_WIDTH / zoom_level))
        painter.drawLine(
            self.position,
            self.position + qt_point(node_basis_display_vector(self, 1)),
        )
        painter.setPen(QPen(Qt.red, BASIS_VECTOR_WIDTH / zoom_level))
        painter.drawLine(
            self.position,
            self.position + qt_point(node_basis_display_vector(self, 2)),
        )

        self.ellipse_item.setPen(QPen(Qt.black, 1./zoom_level))
        self.ellipse_item.paint(painter,option)

        painter.setPen(QPen(Qt.white))
        font_size = int(10 / zoom_level)
        painter.setFont(QFont("Arial",font_size))
   #     painter.drawText(self.position+QPointF(-font_size,font_size/2),str(self.index))


SIMPLE_SAVE_COMPACTION_ERROR = (
    "Saving grids with deleted nodes/elements requires mesh compaction and "
    "renumbering. This is not implemented yet."
)


def live_grid_arrays():
    """Build validated physical grid arrays from the current editor objects."""
    nodes = list(globals().get("node_list", []))
    elements = list(globals().get("element_list", []))
    if any(not getattr(node, "active", True) for node in nodes):
        raise ValueError(SIMPLE_SAVE_COMPACTION_ERROR)
    if any(not getattr(element, "active", True) for element in elements):
        raise ValueError(SIMPLE_SAVE_COMPACTION_ERROR)
    if [node.index for node in nodes] != list(range(len(nodes))):
        raise ValueError(SIMPLE_SAVE_COMPACTION_ERROR)
    if [element.index for element in elements] != list(range(len(elements))):
        raise ValueError(SIMPLE_SAVE_COMPACTION_ERROR)

    nodes_xx = np.zeros((2, 4, len(nodes)), dtype=float)
    boundary = np.empty(len(nodes), dtype=np.int32)
    for index, node in enumerate(nodes):
        nodes_xx[:, 0, index] = np_point(node.position) / this_scaling
        nodes_xx[:, 1:, index] = node.xx[:, 1:] / this_scaling
        boundary[index] = node.boundary

    if elements:
        if any(len(element.vertices) != 4 for element in elements):
            raise ValueError("Every saved grid element must have four vertices")
        vertices = np.asarray(
            [element.vertices for element in elements], dtype=np.int64
        ).T
        element_sizes = np.stack(
            [np.asarray(element.sizes) for element in elements], axis=2
        )
    else:
        vertices = np.empty((4, 0), dtype=np.int64)
        element_sizes = np.empty((4, 4, 0), dtype=float)
    jorek_grid.validate_grid_arrays(
        nodes_xx, boundary, vertices, element_sizes
    )
    return nodes_xx, boundary, vertices, element_sizes


def rebuild_graphics_layers(active_view=None, topology_changed=False):
    """Recompute the configured editable overlay and static mesh."""
    global static_mesh
    if active_view is not None and topology_changed:
        active_view.clear_geometry_undo()
    if scene is None:
        return set(), set()
    if active_view is None:
        active_view = globals().get("view")
    if topology_changed and active_view is not None:
        active_view.invalidate_element_adjacency()
        scene._element_adjacency = None

    boundary_specs = []
    for edge in boundary_list:
        if not getattr(edge, "active", True):
            continue
        element_index = edge.element_index
        element_side = edge.element_side
        if element_index is None:
            continue
        boundary_specs.append((element_index, element_side, edge.vertices))

    if active_view is not None:
        if active_view._element_adjacency is None:
            active_view._element_adjacency = element_edge_adjacency(
                element_list
            )
            scene._element_adjacency = active_view._element_adjacency
        adjacency = active_view._element_adjacency
        depth = active_view.editable_depth
    else:
        adjacency = element_edge_adjacency(element_list)
        scene._element_adjacency = adjacency
        depth = 0
    editable_elements = editable_element_indices(
        element_list, boundary_list, depth=depth, adjacency=adjacency
    )
    editable_nodes = editable_node_indices(element_list, editable_elements)
    old_boundary_edges = set(boundary_list)

    for edge in list(boundary_list):
        if isinstance(edge, QGraphicsItem) and edge.scene() is scene:
            deactivate_boundary_edge(edge)

    for list_index, element in enumerate(list(element_list)):
        should_be_interactive = (
            getattr(element, "active", True)
            and element.index in editable_elements
        )
        if should_be_interactive:
            if not isinstance(element, jorek_element_item):
                old_element = element
                element = jorek_element_item(
                    element.index, element.vertices, element.sizes
                )
                element.edges = [
                    edge for edge in getattr(old_element, "edges", [])
                    if edge not in old_boundary_edges
                ]
                element_list[list_index] = element
            else:
                element.edges = [
                    edge for edge in element.edges
                    if edge not in old_boundary_edges
                ]
            if element.scene() is None:
                scene.addItem(element)
        else:
            if isinstance(element, jorek_element_item):
                if element.scene() is scene:
                    scene.removeItem(element)
                element_list[list_index] = mesh_element_record(
                    element.index, element.vertices, element.sizes,
                    active=getattr(element, "active", True),
                )

    for node_index, node in enumerate(list(node_list)):
        if not getattr(node, "active", True):
            continue
        should_be_interactive = (
            node_index in editable_nodes
        )
        if should_be_interactive:
            if not isinstance(node, jorek_node_item):
                node = jorek_node_item(
                    node.index,
                    this_scaling * jorek.nodes_xx[:, :, node.index],
                    node.boundary,
                )
                node_list[node_index] = node
            if node.scene() is None:
                scene.addItem(node)
        else:
            if isinstance(node, jorek_node_item):
                if node.scene() is scene:
                    scene.removeItem(node)
                node_list[node_index] = mesh_node_record(
                    node.index,
                    this_scaling * jorek.nodes_xx[:, :, node.index],
                    node.boundary,
                    active=getattr(node, "active", True),
                )

    elements_by_index = {
        element.index: element for element in element_list
    }
    new_boundary_list = []
    for element_index, element_side, vertices in boundary_specs:
        element = elements_by_index.get(element_index)
        if element is None or not isinstance(element, jorek_element_item):
            continue
        if element_side is None:
            edge_key = frozenset(vertices)
            element_side = next(
                side for side in range(4)
                if element_side_key(element, side) == edge_key
            )
        edge = boundary_edge_from_element_side(element, element_side)
        element.edges.append(edge)
        new_boundary_list.append(edge)
        scene.addItem(edge)
    boundary_list[:] = new_boundary_list

    rebuild_node_connections()
    if not graphics_item_in_scene(static_mesh, scene):
        static_mesh = static_mesh_item()
        scene.addItem(static_mesh)
    else:
        static_mesh.rebuild_path()

    if active_view is not None:
        active_view.editable_element_indices_set = set(editable_elements)
        active_view.editable_node_indices_set = set(editable_nodes)
        active_view.selected_nodes = []
        active_view.selected_elements = []
        active_view.selected_edges = []
        active_view.selected_point = None
        active_view.dragged_node = None
    scene.update()
    return editable_elements, editable_nodes


def build_grid_scene(grid, scaling, editable_depth=0):
    """Construct a complete replacement scene from validated grid arrays."""
    global jorek, scene, node_list, element_list, boundary_list, static_mesh
    jorek_grid.validate_grid_arrays(
        grid.nodes_xx, grid.boundary, grid.vertices, grid.elements_size
    )
    previous = (
        globals().get("jorek"), scene, node_list, element_list, boundary_list,
        static_mesh,
    )
    try:
        jorek = grid
        scene = QGraphicsScene()
        scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        if MEMORY_DIAGNOSTICS:
            print("[MEM] scene index method: NoIndex", flush=True)
        node_list = [
            mesh_node_record(
                index, scaling * grid.nodes_xx[:, :, index],
                grid.boundary[index],
            )
            for index in range(grid.nodes_xx.shape[2])
        ]
        report_memory("after node construction")
        element_list = [
            mesh_element_record(
                index, grid.vertices[:, index],
                grid.elements_size[:, :, index],
            )
            for index in range(grid.vertices.shape[1])
        ]
        report_memory("after element construction")
        boundary_sides = []
        for element in element_list:
            for element_side in range(4):
                vertex0 = element.vertices[element_side]
                vertex1 = element.vertices[(element_side + 1) % 4]
                if node_list[vertex0].boundary and node_list[vertex1].boundary:
                    boundary_sides.append((element.index, element_side))

        adjacency = element_edge_adjacency(element_list)
        scene._element_adjacency = adjacency
        editable_elements = expand_editable_element_indices(
            {index for index, unused_side in boundary_sides},
            adjacency,
            editable_depth,
        )
        editable_nodes = editable_node_indices(element_list, editable_elements)
        for node_index in editable_nodes:
            record = node_list[node_index]
            node_list[node_index] = jorek_node_item(
                record.index, record.xx, record.boundary
            )
        for list_index, record in enumerate(element_list):
            if record.index in editable_elements:
                element_list[list_index] = jorek_element_item(
                    record.index, record.vertices, record.sizes
                )

        elements_by_index = {
            element.index: element for element in element_list
        }
        boundary_list = []
        for element_index, element_side in boundary_sides:
            element = elements_by_index[element_index]
            edge = boundary_edge_from_element_side(element, element_side)
            boundary_list.append(edge)
            element.edges.append(edge)
        report_memory("after boundary discovery")
        report_boundary_diagnostics()
        report_graphics_multiplication()

        static_mesh = static_mesh_item()
        scene.addItem(static_mesh)
        for edge in boundary_list:
            scene.addItem(edge)
        report_memory("after boundary insertion")
        for node in node_list:
            if isinstance(node, jorek_node_item):
                scene.addItem(node)
        report_memory("after node insertion")
        for element in element_list:
            if isinstance(element, jorek_element_item):
                scene.addItem(element)
        report_memory("after element insertion")
        rebuild_node_connections()
        report_memory("after rebuild connections")
        return scene, node_list, element_list, boundary_list
    except Exception:
        (
            jorek, scene, node_list, element_list, boundary_list, static_mesh,
        ) = previous
        raise


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """Generate a short random key."""
    return ''.join(random.choice(chars) for x in range(size))
if __name__ == "__main__":
    app   = QApplication(sys.argv)
    if MEMORY_DIAGNOSTICS:
        reset_memory_diagnostics()
        report_memory("application started")
    view  = this_view()
    scene = QGraphicsScene()
    scene.setItemIndexMethod(QGraphicsScene.NoIndex)
    view.setScene(scene)
    window = grid_editor_window(view)
    window.show()
    if len(sys.argv) > 1:
        window.open_grid_file(sys.argv[1], interactive=True)
    else:
        view.set_patch_status("Use File > Open to load a grid")
    if MEMORY_DIAGNOSTICS:
        app.processEvents()
        report_memory("after first render")
        print_memory_diagnostic_table()
        if os.environ.get(
            "JOREK_GRID_MEMORY_DIAGNOSTIC_EXIT", "0"
        ) == "1":
            QTimer.singleShot(0, app.quit)
    elif DIAGNOSTIC_BASIS_SCALE:
        app.processEvents()
    if DIAGNOSTIC_BASIS_SCALE and len(sys.argv) > 1:
        verify_diagnostic_grid_unchanged(jorek, "after first render")
    sys.exit(app.exec_())
