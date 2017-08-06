import numpy as np
import networkx as nx
import sys

INTMAX = sys.maxsize


def find_graph_cut(imlr, imrl, imrr, imll, maskSize, xoffsetL, xoffsetR):
    mask_l = get_single_mask(imlr, imrl)
    mask_r = get_single_mask(imrr, imll)
    mask_all = np.zeros((maskSize[1], maskSize[0], 3), np.float64)
    mask_all[:, :xoffsetL, :] = 1
    mask_all[:, xoffsetR + mask_r.shape[1]:, :] = 1
    mask_all[:, xoffsetL:xoffsetL + mask_l.shape[1], :] = mask_l
    mask_all[:, xoffsetR:xoffsetR + mask_r.shape[1], :] = 1 - mask_r
    return mask_all


def get_single_mask(i_l, i_r):
    h, w = i_l.shape[:2]

    L2_NORM = np.linalg.norm(i_l - i_r, axis=2)

    G1 = build_graph(i_l, i_r, L2_NORM)

    cut_value, partition = nx.minimum_cut(G1, str(h * w), str(h * w + 1))
    reachable, non_reachable = partition

    mask = np.zeros((1, h * w))
    reachable_l = list(map(int, reachable))
    reachable_l.remove(h * w)
    mask[0][reachable_l] = 1
    mask = mask.reshape((h, w))
    mask_color = np.zeros((h, w, 3))
    mask_color[:, :, 0] = mask
    mask_color[:, :, 1] = mask
    mask_color[:, :, 2] = mask

    return mask_color


def build_graph(im_a, im_b, L2_NORM):
    h, w = im_a.shape[:2]
    G = nx.Graph()
    G.add_nodes_from(np.char.mod('%d', np.arange(h * w + 2)))

    idx_source = h * w
    idx_sink = h * w + 1

    indices = np.arange(h * w).reshape((h, w))
    cost_ver_edge = L2_NORM[:, :-1] + L2_NORM[:, 1:]
    cost_hor_edge = L2_NORM[:-1] + L2_NORM[1:]

    dict_lr = [(str(x), str(y), {'capacity': z}) for x, y, z in zip(
        indices[:, :-1].ravel(), indices[:, 1:].ravel(),
        cost_ver_edge.ravel())]
    dict_bt = [(str(x), str(y), {'capacity': z}) for x, y, z in zip(
        indices[:-1].ravel(), indices[1:].ravel(), cost_hor_edge.ravel())]
    G.add_edges_from(dict_lr)
    G.add_edges_from(dict_bt)

    # left most column infinity
    dict_left_most = [(str(x), str(idx_source), {'capacity': INTMAX})
                      for x in indices[:, 0].ravel()]
    G.add_edges_from(dict_left_most)
    # print (dict_left_most)

    # right most column infinity
    dict_right_most = [(str(x), str(idx_sink), {'capacity': INTMAX})
                       for x in indices[:, w - 1].ravel()]
    G.add_edges_from(dict_right_most)

    return G
