#!/usr/bin/python
import numpy as np
import cv2


def y_align(img, pxs):
    l = img[:, :1280, :]
    r = img[:, 1280:, :]

    region_l = l[1280 - pxs:, :, :]
    region_r = r[:pxs, :, :]
    region_l = cv2.flip(region_l, -1)
    region_r = cv2.flip(region_r, -1)

    l_result = np.append(region_r, l[:1280 - pxs, :, :], axis=0)
    r_result = np.append(r[pxs:, :, :], region_l, axis=0)
    result = np.append(l_result, r_result, axis=1)

    return result


img = cv2.imread('extract.png', cv2.IMREAD_COLOR)
result = y_align(img, 5)
cv2.imwrite('extract_aligned.png', result)
