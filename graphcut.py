import cv2
import numpy as np
import networkx as nx
import sys

INTMAX = sys.maxsize
K = 5


def find_graph_cut(imset_t, maskSize, xoffsetL, xoffsetR,
                   imset_t_1=None, mask_old=None):
    imlr_t, imrl_t, imrr_t, imll_t = imset_t
    if imset_t_1 is None:
        mask_l = get_single_mask(imlr_t, imrl_t)
        mask_r = get_single_mask(imrr_t, imll_t)
    else:
        imlr_t_1, imrl_t_1, imrr_t_1, imll_t_1 = imset_t_1
        mask_l = get_single_mask(
            imlr_t, imrl_t, imlr_t_1, imrl_t_1,
            mask_old[:, xoffsetL:xoffsetL + imlr_t_1.shape[1]])
        mask_r = get_single_mask(
            imrr_t, imll_t, imrr_t_1, imll_t_1,
            1 - mask_old[:, xoffsetR:xoffsetR + imrr_t_1.shape[1]])
    mask_all = np.zeros((maskSize[1], maskSize[0], 3), np.float64)
    mask_all[:, :xoffsetL] = 1
    mask_all[:, xoffsetR + mask_r.shape[1]:] = 1
    mask_all[:, xoffsetL:xoffsetL + mask_l.shape[1]] = mask_l
    mask_all[:, xoffsetR:xoffsetR + mask_r.shape[1]] = 1 - mask_r

    cv2.imshow('mask', mask_all[:, :, 0].astype(np.float32))
    return mask_all


def get_single_mask(imgA_t, imgB_t,
                    imgA_t_1=None, imgB_t_1=None, smask_old=None):
    h, w = imgA_t.shape[:2]
    L2_NORM_t = np.linalg.norm(imgA_t - imgB_t, axis=2)
    if smask_old is None:
        G = build_graph(h, w, [L2_NORM_t])
        cut_value, partition = nx.minimum_cut(G, str(h * w), str(h * w + 1))
    else:
        L2_NORM_AA = np.linalg.norm(imgA_t - imgA_t_1, axis=2)
        L2_NORM_BB = np.linalg.norm(imgB_t - imgB_t_1, axis=2)
        L2_NORM_AB = np.linalg.norm(imgA_t - imgB_t_1, axis=2)
        L2_NORM_BA = np.linalg.norm(imgB_t - imgA_t_1, axis=2)
        G = build_graph(h, w, [L2_NORM_t, L2_NORM_AA +
                               L2_NORM_BB + L2_NORM_AB + L2_NORM_BA],
                        smask_old)
        cut_value, partition = nx.minimum_cut(
            G, str(2 * h * w), str(2 * h * w + 1))

    reachable, non_reachable = partition
    mask = np.zeros((h * w,))
    reachable = np.int32(list(reachable))
    mask[reachable[reachable < h * w]] = 1
    mask = mask.reshape((h, w))
    mask_color = np.zeros((h, w, 3))
    mask_color[:, :, 0] = mask
    mask_color[:, :, 1] = mask
    mask_color[:, :, 2] = mask

    return mask_color


def build_graph(h, w, L2_NORM, smask_old=None):
    G = nx.Graph()
    indices = np.arange(h * w).reshape((h, w))

    if len(L2_NORM) == 1:
        G.add_nodes_from(np.char.mod('%d', np.arange(h * w + 2)))
        idx_source = h * w
        idx_sink = h * w + 1
    elif len(L2_NORM) == 2:
        G.add_nodes_from(np.char.mod('%d', np.arange(2 * h * w + 2)))
        idx_source = 2 * h * w
        idx_sink = 2 * h * w + 1

        # data penalty of t-1
        dict_dp_t_1 = [(str(x + h * w), str(idx_source), {'capacity': INTMAX})
                       for x in indices[smask_old == 1].ravel()]
        G.add_edges_from(dict_dp_t_1)
        dict_dp_t_1 = [(str(x + h * w), str(idx_sink), {'capacity': INTMAX})
                       for x in indices[smask_old == 0].ravel()]
        G.add_edges_from(dict_dp_t_1)

        # temporal coherence
        dict_tc = [(str(x), str(x + h * w), {'capacity': tc * K})
                   for x, tc in zip(indices.ravel(), L2_NORM[1].ravel())]
        G.add_edges_from(dict_tc)

    # data penalty of t
    dict_dp_t = [(str(x), str(idx_source), {'capacity': INTMAX})
                 for x in indices[:, 0].ravel()]
    G.add_edges_from(dict_dp_t)
    dict_dp_t = [(str(x), str(idx_sink), {'capacity': INTMAX})
                 for x in indices[:, -1].ravel()]
    G.add_edges_from(dict_dp_t)

    # spatial coherence of t
    cost_hor_edge = L2_NORM[0][:, :-1] + L2_NORM[0][:, 1:]
    cost_ver_edge = L2_NORM[0][:-1] + L2_NORM[0][1:]

    dict_hor = [(str(x), str(y), {'capacity': z}) for x, y, z in zip(
        indices[:, :-1].ravel(), indices[:, 1:].ravel(),
        cost_hor_edge.ravel())]
    dict_ver = [(str(x), str(y), {'capacity': z}) for x, y, z in zip(
        indices[:-1].ravel(), indices[1:].ravel(), cost_ver_edge.ravel())]
    G.add_edges_from(dict_hor)
    G.add_edges_from(dict_ver)

    return G
