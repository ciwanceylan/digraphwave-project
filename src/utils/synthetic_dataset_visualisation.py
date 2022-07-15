from typing import Dict, List
import numpy as np
import graph_tool.all as gt
import matplotlib.colors
import copy
from sklearn.decomposition import PCA

from src.utils.synthetic_dataset import SyntheticGraph

colors_cb_bw = {
    "seq": ["#fef0d9", "#fdcc8a", "#fc8d59", "#d7301f"],
    "div": ["#e66101", "#fdb863", "#b2abd2", "#5e3c99"],
    "qual": ["#1b9e77", "#d95f02", "#7570b3", "#000000"]
}

colors_cb = {
    "seq_warm": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"],
    "seq_cold": ["#fff7fb", "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#045a8d", "#023858"],
    "div": ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4",
            "#313695"]
}

graph_tool_shapes = ["circle", "triangle", "square", "pentagon", "double_circle", "hexagon", "double_triangle",
                     "heptagon", "double_square", "octagon", "double_hexagon",
                     "double_heptagon",
                     "double_octagon", "pie", "circle"]


class VisualisationInfo:
    predefined_pos: Dict[str, np.ndarray]
    predefined_node2color: Dict[str, Dict[int, List[float]]]

    def __init__(self):
        self.predefined_pos = {
            "B5": B5_pos(),
            "C8": C8_pos(),
            "U5": U5_pos(),
            "H5": H5_pos(),
            "D5": D5_pos(),
            "PB5": PB5_pos(),
            "L5": L5_pos(),
            "DB55": DB55_pos(),
            "S5": S5_pos(),
            "W5": W5_pos(),
            "DS5": DS5_pos(),
        }
        self.predefined_node2color = dict()

    def get_pos(self, graph: SyntheticGraph, transform=True):
        if graph.core_name in self.predefined_pos:
            pos = self.predefined_pos[graph.core_name]
            pos = pos.T
        else:
            pos = gt.sfdp_layout(graph.gt_graph).get_2d_array(pos=[0, 1])
            self.predefined_pos[graph.core_name] = pos
            pos = pos.T
            if transform:
                pos = PCA(n_components=2).fit_transform(pos)

        pos = pos - np.mean(pos, axis=0)
        pos = pos / np.abs(pos).max()

        return pos

    def get_pos_by_core_name(self, core_name: str):
        pos = self.predefined_pos[core_name]
        pos = pos.T
        pos = pos - np.mean(pos, axis=0)
        pos = pos / np.abs(pos).max()
        return pos

    def get_color_map(self, graph: SyntheticGraph):
        if graph.name in self.predefined_node2color:
            node2color = self.predefined_node2color[graph.name]
        else:
            node2color = get_colors_from_gtgraph(graph.gt_graph)
            self.predefined_node2color[graph.name] = node2color
        return node2color


def get_colors_from_gtgraph(gt_graph: gt.Graph):
    colors = copy.deepcopy(colors_cb['div'])
    labels = gt_graph.vp["label"]
    # node2color = graph.gt_graph.new_vp('vector<float>')
    node2color = dict()
    max_label = labels.a.max()
    if len(colors) < max_label:
        colors.append("#000000")
    if len(colors) < max_label:
        colors.append("#9ec46d")

    for node in gt_graph.vertices():
        if labels[node] < 0:
            node2color[node] = [0., 0., 0.]
        else:
            hex_color = colors[int((len(colors) - 1) * labels[node] / max_label)]
            node2color[node] = list(matplotlib.colors.to_rgb(hex_color))
    return node2color


def B5_pos():
    pos = np.asarray(
        [
            [-5.60610401, -1.09568664],
            [-6.43723406, -1.29506959],
            [-6.8805207, -0.52347291],
            [-6.24813379, 0.06772526],
            [-5.37821297, -0.15122509],
            [-3.56463847, 0.77644666],
            [-1.81502781, 1.39554479],
            [-0.00982275, 1.61749376],
            [1.80052609, 1.41703506],
            [3.56016268, 0.8005211],
            [5.3756363, -0.14802576],
            [6.30458906, 0.06950008],
            [5.60399502, -1.060545],
            [6.88121347, -0.5563009],
            [6.41357193, -1.31394081]
        ]
    )
    ball1 = pos[:5, :]
    ball1_mean = np.mean(ball1, axis=0)
    ball2 = pos[-5:, :]
    ball2_mean = np.mean(ball2, axis=0)
    new_ball_pos1 = 1.75 * (ball1 - ball1_mean) + ball1_mean
    new_ball_pos2 = 1.75 * (ball2 - ball2_mean) + ball2_mean
    pos[:4, :] = new_ball_pos1[:4, :]
    pos[-4:, :] = new_ball_pos2[-4:, :]
    return pos.T


def L5_pos():
    pos = B5_pos()
    return pos[:, :-5]


def C8_pos():
    r = 0.1
    pos = np.zeros((2, 8))
    for i in range(8):
        pos[0, i] = r * np.cos(2 * np.pi * i / 8 + np.pi)
        pos[1, i] = r * np.sin(2 * np.pi * i / 8 + np.pi)
    return pos


def U5_pos():
    pos = [
        [0, 2],
        [0, 0],
    ]
    for i in range(5):
        angle = (2 * np.pi * i / 6) - (np.pi / 6)
        pos.append(
            [2 * np.cos(angle),
             -2 * np.sin(angle)]
        )
    return 0.5 * np.asarray(pos).T


def H5_pos():
    pos = np.asarray([
        [0, -0.5],
        [-1, 0],
        [1, 0],
        [-1, 1],
        [1, 1]
    ])
    return 0.75 * pos.T


def D5_pos():
    pos = [[0, -1]]
    for i in range(2):
        for j in range(5):
            pos.append([i + 1, -0.5 * j])
    pos.append([3, -1])

    return 0.75 * np.asarray(pos).T


def PB5_pos():
    pos = np.asarray([
        [2, 1],
        [3.5, 1],
        [3.5, -1],
        [2, -1],
        [0.75, 0],
        [-0.75, 0],
        [-2, -1],
        [-3.5, -1],
        [-3.5, 1],
        [-2, 1]
    ])
    # pos[:, 0] = 0.8 * pos[:, 0]
    return pos.T


def DB55_pos():
    pos = np.asarray([
        [-2.05498147, -0.33561963],
        [-1.44680739, -0.4946154],
        [-1.45070285, 0.49631368],
        [-2.06228367, 0.33011699],
        [-0.90643266, 0.00352821],
        [0.90955661, 0.00273465],
        [1.43115269, -0.49578936],
        [2.07313334, 0.33086701],
        [1.43631938, 0.49887189],
        [2.07104601, -0.33640803]
    ])
    pos[:, 0] = 0.75 * pos[:, 0]
    return pos.T


def S5_pos():
    r = 0.5
    pos = np.zeros((2, 6))
    for i in range(1, 6):
        pos[0, i] = r * np.cos(2 * np.pi * i / 5 + np.pi)
        pos[1, i] = r * np.sin(2 * np.pi * i / 5 + np.pi)
    return pos


def DS5_pos():
    r = 0.75
    s1 = S5_pos()
    s1[0, :] -= r
    s2 = np.asarray([[np.cos(np.pi / 5), -np.sin(np.pi / 5)], [np.sin(np.pi / 5), np.cos(np.pi / 5)]]) @ S5_pos()
    s2[0, :] += r
    return np.concatenate((s1, s2), axis=1)


def W5_pos():
    pos = np.asarray([
        [9.08606216e-01, -9.17916946e-01],
        [7.08797639e-01, 2.51700892e-03],
        [9.13705996e-01, 9.17446163e-01],
        [2.12947725e+00, 6.13985552e-01],
        [2.12616059e+00, -6.17563738e-01],
        [1.55948347e+00, -6.74392269e-04],
        [-7.14103369e-01, 8.50852345e-04],
        [-9.18168773e-01, -9.21345665e-01],
        [-2.11927568e+00, -6.16720691e-01],
        [-2.12536937e+00, 6.15987095e-01],
        [-9.11869852e-01, 9.22654210e-01],
        [-1.55744411e+00, 7.80552185e-04],
    ])
    return pos.T


def edge_end_markers(gt_graph: gt.Graph):
    end_marker = gt_graph.new_ep('string')
    for e in gt_graph.edges():
        if not gt_graph.edge(e.target(), e.source()):
            end_marker[e] = "arrow"
        else:
            end_marker[e] = "none"
    return end_marker


def draw(graph: SyntheticGraph, output=None, vis_info: VisualisationInfo = None, transform: bool = True):
    if vis_info is None:
        vis_info = VisualisationInfo()
    pos = vis_info.get_pos(graph, transform=transform)
    node2color = vis_info.get_color_map(graph)

    return draw_gt_graph(graph.gt_graph, pos, node2color, output=output)


def get_pos_composed_graph(graph: gt.Graph):
    vis_info = VisualisationInfo()
    pos = np.zeros((graph.num_vertices(), 2))
    subgraph_changes = np.diff(graph.vp.subgraph_label.a, prepend=-2).astype(bool)
    names = [str_label.split("_")[0] for i, str_label in enumerate(graph.vp.str_label) if subgraph_changes[i]]
    num_main_nodes = (graph.vp.subgraph_label.a == -1).sum()
    for s, name in enumerate(names):
        mask = graph.vp.subgraph_label.a == s - 1

        if name != "main":
            sub_graph_pos = vis_info.get_pos_by_core_name(name)
            pos[mask, :] = transform_subgraph_pos(sub_graph_pos, s, num_main_nodes)
        else:
            main_graph_pos = get_main_graph_pos(num_main_nodes)
            pos[mask, :] = main_graph_pos

    pos = pos - np.mean(pos, axis=0)
    pos = pos / np.abs(pos).max()

    return pos


def get_main_graph_pos(num_nodes):
    radius = num_nodes / (5 * np.pi)
    angles = 2 * np.pi * np.arange(num_nodes) / num_nodes
    pos = radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return pos


def transform_subgraph_pos(subgraph_pos, s_val, num_main_nodes):
    radius_main = num_main_nodes / (5 * np.pi)
    rot_angle = (- 2 * np.pi * 5 * s_val / num_main_nodes) + (2 * np.pi * 3.5 / num_main_nodes)
    rot_mat = np.asarray([[np.sin(rot_angle), np.cos(rot_angle)],
                          [-np.cos(rot_angle), np.sin(rot_angle)]])
    shift = 1.5 * radius_main * np.asarray([[np.cos(rot_angle), -np.sin(rot_angle)]])
    subgraph_pos = shift + (subgraph_pos @ rot_mat)

    return subgraph_pos


def draw_composed_graph(composed_graph: gt.Graph, scale: float, output: str = None):
    pos = get_pos_composed_graph(composed_graph)
    # node2color = get_colors_from_gtgraph(composed_graph)
    node2color = None
    return draw_gt_graph(composed_graph, pos=pos, node2color=node2color, scale=scale, output=output, label_nodes=False)


def color_contrast(color):
    c = np.asarray(color)
    y = c[0] * .299 + c[1] * .587 + c[2] * .114
    if y < .5:
        c[:3] = 1
    else:
        c[:3] = 0
    return c


def calc_color_y(c):
    y = c[0] * .299 + c[1] * .587 + c[2] * .114
    return y


def draw_gt_graph(gt_graph: gt.Graph, pos: np.ndarray = None, node2color: Dict = None, output=None,
                  scale: float = 6, label_nodes: bool = True):
    if pos is None:
        pos = gt.sfdp_layout(gt_graph).get_2d_array(pos=[0, 1]).T
        pos = pos - np.mean(pos, axis=0)
        pos = pos / np.abs(pos).max()

    margin_factor = 1.25
    max_x, max_y = np.max(np.abs(pos), axis=0)  # One of these should be 1.
    view_tuple = (-margin_factor * max_x, -margin_factor * max_y, 2 * margin_factor * max_x, 2 * margin_factor * max_y)
    output_size = (300 * max_x / max_y, 300)

    pos_vp = gt_graph.new_vp('vector<float>')
    for node, pos_ in enumerate(pos):
        pos_vp[node] = pos_

    node2color_vp = None
    node2edgecolor_vp = None
    node2textcolor_vp = None
    shape_vp = None
    vertex_text = None

    if label_nodes:
        vertex_text = gt_graph.vp["label"]

    # if node2color is None:
    #     node2color = get_colors_from_gtgraph(gt_graph)

    if node2color is not None:
        node2color_vp = gt_graph.new_vp('vector<float>')
        node2edgecolor_vp = gt_graph.new_vp('vector<float>')
        node2textcolor_vp = gt_graph.new_vp('vector<float>')
        for node, color in node2color.items():
            node2color_vp[node] = color
            color_amp = (1 - calc_color_y(color)) / 0.5
            node2edgecolor_vp[node] = color_amp * np.asarray(color)
            # node2textcolor_vp[node] = np.asarray(3*[1 - np.median(color)])
            node2textcolor_vp[node] = color_contrast(color)

        shape_vp = gt_graph.new_vp('string')
        for node in gt_graph.vertices():
            shape_vp[node] = graph_tool_shapes[gt_graph.vp.label[node]]

    end_markers = edge_end_markers(gt_graph)
    control_points = gt_graph.new_ep('vector<float>')

    gt.graph_draw(gt_graph,
                  pos=pos_vp,
                  vertex_fill_color=node2color_vp,
                  vertex_color=node2edgecolor_vp,
                  vertex_text=vertex_text,
                  vertex_text_color=node2textcolor_vp,
                  vertex_size=10 * scale,
                  vertex_shape=shape_vp,
                  edge_pen_width=1 * scale,
                  edge_color='k',
                  edge_marker_size=5 * scale,
                  edge_end_marker=end_markers,
                  edge_control_points=control_points,
                  output=output,
                  fit_view=view_tuple,
                  output_size=output_size
                  )

    return pos
