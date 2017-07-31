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


def imgLabeling2(img1, img2, img3, img4, xoffsetL, xoffsetR):
    errL = np.sum(np.square(img1.astype(np.float64) -
                            img2.astype(np.float64)), axis=2)
    errR = np.sum(np.square(img3.astype(np.float64) -
                            img4.astype(np.float64)), axis=2)
    EL = np.zeros(errL.shape, np.float64)
    ER = np.zeros(errR.shape, np.float64)
    EL[0] = errL[0]
    ER[0] = errR[0]
    for i in range(1, 1280):
        EL[i, 0] = errL[i, 0] + min(EL[i - 1, 0], EL[i - 1, 1])
        ER[i, 0] = errR[i, 0] + min(ER[i - 1, 0], ER[i - 1, 1])
        for j in range(1, EL.shape[1] - 1):
            EL[i, j] = errL[i, j] + \
                min(EL[i - 1, j - 1], EL[i - 1, j], EL[i - 1, j + 1])
            ER[i, j] = errR[i, j] + \
                min(ER[i - 1, j - 1], ER[i - 1, j], ER[i - 1, j + 1])
        EL[i, -1] = errL[i, -1] + min(EL[i - 1, -1], EL[i - 1, -2])
        ER[i, -1] = errR[i, -1] + min(ER[i - 1, -1], ER[i - 1, -2])

    minlocL = np.argmin(EL, axis=1) + xoffsetL
    minlocR = np.argmin(ER, axis=1) + xoffsetR
    mask = np.ones((1280, 2560, 3), np.float64)
    for i in range(1280):
        mask[i, minlocL[i]:minlocR[i]] = 0
        mask[i, minlocL[i]] = 0.5
        mask[i, minlocR[i]] = 0.5
    return mask


def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP


def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        LP.append(img - cv2.pyrUp(next_img, img.shape[1::-1]))
        img = next_img
    LP.append(img)
    return LP


def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended


def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, lev_img.shape[1::-1])
        img += lev_img
    return img


def multi_band_blending(img1, img2, mask, leveln=6):
    max_leveln = int(np.floor(
        np.log2(min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print "warning: inappropriate number of leveln"
        leveln = max_leveln


    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(img1.astype(np.float64), leveln)
    LPB = LaplacianPyramid(img2.astype(np.float64), leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result
