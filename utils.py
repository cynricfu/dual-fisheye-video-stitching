#!/usr/bin/python
import numpy as np
import cv2
import sys


def get_theta_phi_2(x_proj, y_proj, W, H, fov):
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * np.pi / H

    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = np.cos(theta_alt) * np.cos(phi_alt)

    return np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)


def buildmap_2(Ws, Hs, Wd, Hd, fov=180.0):
    fov = fov * np.pi / 180.0

    # cartesian coordinates of the projected (square) image
    ys, xs = np.indices((Hs, Ws), np.float32)
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0

    # spherical coordinates
    theta, phi = get_theta_phi_2(x_proj, y_proj, Ws, Hs, fov)

    # polar coordinates (of the fisheye image)
    p = Hd * phi / fov

    # cartesian coordinates of the fisheye image
    y_fish = p * np.sin(theta)
    x_fish = p * np.cos(theta)

    ymap = Hd / 2.0 - y_fish
    xmap = Wd / 2.0 + x_fish
    return xmap, ymap


def getMatches_templmatch(img1, img2, templ_shape, max):
    if not np.array_equal(img1.shape, img2.shape):
        print "error: inconsistent array dimention", img1.shape, img2.shape
        sys.exit()
    if not (np.all(templ_shape <= img1.shape[:2]) and np.all(templ_shape <= img2.shape[:2])):
        print "error: template shape shall fit img1 and img2"
        sys.exit()

    Hs, Ws = img1.shape[:2]
    Ht, Wt = templ_shape
    matches = []
    for yt in range(0, Hs - Ht + 1, 8):
        for xt in range(0, Ws - Wt + 1):
            result = cv2.matchTemplate(
                img1, img2[yt:yt + Ht, xt:xt + Wt], cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            if maxVal > 0.9:
                matches.append((maxVal, maxLoc, (xt, yt)))
    matches.sort(key=lambda e: e[0], reverse=True)
    if len(matches) >= max:
        return np.int32([matches[i][1:] for i in range(max)])
    else:
        return np.int32([c[1:] for c in matches])


def imgLabeling(img1, img2, img3, img4, xoffsetL, xoffsetR):
    minlocL = np.argmin(np.sum(np.square(img1 - img2), axis=2), axis=1)
    minlocR = np.argmin(np.sum(np.square(img3 - img4), axis=2), axis=1)
    minlocL = minlocL + xoffsetL
    minlocR = minlocR + xoffsetR
    mask = np.zeros((1280, 2560, 3), np.float64)
    for i in range(1280):
        mask[i, minlocL[i]:minlocR[i]] = 1
        mask[i, minlocL[i]] = 0.5
        mask[i, minlocR[i]] = 0.5
    return mask
