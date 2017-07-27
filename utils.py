#!/usr/bin/python
import numpy as np
import cv2
import sys


def get_theta_phi(x_proj, y_proj, W, H, fov):
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * np.pi / H

    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = -np.cos(theta_alt) * np.cos(phi_alt)

    return np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)


def buildmap_1(Ws, Hs, Wd, Hd, fov=180.0):
    fov = fov * np.pi / 180.0
    R_max = np.sin(fov / 2.0) / (1 + np.cos(fov / 2.0))

    # cartesian coordinates of the projected (square) image
    ys, xs = np.indices((Hs, Ws), np.float32)
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0

    # spherical coordinates
    theta, phi = get_theta_phi(x_proj, y_proj, Ws, Hs, fov)

    # polar coordinates (of the fisheye image)
    R = np.sin(phi) / (1 - np.cos(phi))

    # cartesian coordinates of the fisheye image
    y_fish = R * np.sin(theta)
    x_fish = R * np.cos(theta)

    ymap = (Hd - y_fish * Hd / R_max) / 2.0
    xmap = (Wd + x_fish * Wd / R_max) / 2.0
    return xmap, ymap


def getMatches_templmatch(img1, img2, templ_shape, max):
    if not np.array_equal(img1.shape, img2.shape):
        print "error: inconsistent array dimention", img1.shape, img2.shape
        sys.exit()
    if not (np.all(templ_shape <= img1.shape[:2]) and np.all(templ_shape <= img2.shape[:2])):
        print "error: template shape shall fit img1 and img2"
        sys.exit()
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
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
        return np.float32([matches[i][1:] for i in range(max)])
    else:
        return np.float32([c[1:] for c in matches])


def getMatches_SIFT_FLANN(img1, img2, ratio, max):
    # detect and extract features from the image
    sift = cv2.xfeatures2d.SIFT_create()
    kps1, des1 = sift.detectAndCompute(img1, None)
    kps2, des2 = sift.detectAndCompute(img2, None)

    # convert the keypoints from KeyPoint objects to NumPy arrays
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # compute the raw matches
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    rawMatches = flann.knnMatch(des1, des2, k=2)

    # perform Lowe's ratio test to get actual matches
    matches = []
    for m, n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < ratio * n.distance:
            # here queryIdx corresponds to kps1
            # trainIdx corresponds to kps2
            matches.append((kps1[m.queryIdx], kps2[m.trainIdx]))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # return the matches
        return np.float32(matches)
    else:
        # otherwise, no homograpy could be computed
        return None


def getMatches_SIFT_BF(img1, img2, ratio, max):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kps1, des1 = sift.detectAndCompute(img1, None)
    kps2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    rawMatches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    matches = []
    for m, n in rawMatches:
        if m.distance < ratio * n.distance:
            matches.append((kps1[m.queryIdx].pt, kps2[m.trainIdx].pt))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # return the matches
        return np.float32(matches)
    else:
        # otherwise, no homograpy could be computed
        return None
