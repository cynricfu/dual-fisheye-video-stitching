#!/usr/bin/python
import numpy as np


def verticalBoundary(M, W_remap, W, H):
    """Return vertical boundary of input image."""
    row = np.zeros((W_remap, 3, 1))
    row[:, 2] = 1
    row[:, 0] = np.arange(
        (W - W_remap) / 2, (W + W_remap) / 2).reshape((W_remap, 1))
    product = np.matmul(M, row).reshape((W_remap, 3))
    normed = np.array(
        zip(product[:, 0] / product[:, 2], product[:, 1] / product[:, 2]))
    top = np.max(normed[np.logical_and(normed[:, 0] >= W_remap / 2,
                                       normed[:, 0] < W - W_remap / 2)][:, 1])

    row[:, 1] = H - 1
    product = np.matmul(M, row).reshape((W_remap, 3))
    normed = np.array(
        zip(product[:, 0] / product[:, 2], product[:, 1] / product[:, 2]))
    bottom = np.min(normed[np.logical_and(
        normed[:, 0] >= W_remap / 2, normed[:, 0] < W - W_remap / 2)][:, 1])

    return int(top) if top > 0 else 0, int(bottom) if bottom < H else H
