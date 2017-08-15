#!/usr/bin/python
import numpy as np
import cv2
import sys


def getMatches_templmatch(img1, img2, templ_shape, max):
    """Return pairs of corresponding points
    using brute force template matching."""
    if not np.array_equal(img1.shape, img2.shape):
        print "error: inconsistent array dimention", img1.shape, img2.shape
        sys.exit()
    if not (np.all(templ_shape <= img1.shape[:2]) and
            np.all(templ_shape <= img2.shape[:2])):
        print "error: template shape shall fit img1 and img2"
        sys.exit()

    Hs, Ws = img1.shape[:2]
    Ht, Wt = templ_shape
    matches = []
    for yt in range(0, Hs - Ht + 1, 64):
        for xt in range(0, Ws - Wt + 1, 2):
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


def getMatches_goodtemplmatch(img1, img2, templ_shape, max):
    """Return pairs of corresponding points
    using combining Shi-Tomasi corner detector and template matching."""
    if not np.array_equal(img1.shape, img2.shape):
        print "error: inconsistent array dimention", img1.shape, img2.shape
        sys.exit()
    if not (np.all(templ_shape <= img1.shape[:2]) and
            np.all(templ_shape <= img2.shape[:2])):
        print "error: template shape shall fit img1 and img2"
        sys.exit()

    feature_params = dict(maxCorners=max, qualityLevel=0.01,
                          minDistance=5, blockSize=5)
    kps1 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    kps2 = cv2.goodFeaturesToTrack(img2, mask=None, **feature_params)

    Hs, Ws = img1.shape[:2]
    Ht, Wt = templ_shape
    matches = []
    for [[xt, yt]] in kps1:
        if int(yt) + Ht > Hs or int(xt) + Wt > Ws:
            continue
        result = cv2.matchTemplate(
            img2, img1[int(yt):int(yt) + Ht, int(xt):int(xt) + Wt],
            cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal > 0.85:
            matches.append((maxVal, (int(xt), int(yt)), maxLoc))
    for [[xt, yt]] in kps2:
        if int(yt) + Ht > Hs or int(xt) + Wt > Ws:
            continue
        result = cv2.matchTemplate(
            img1, img2[int(yt):int(yt) + Ht, int(xt):int(xt) + Wt],
            cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal > 0.85:
            matches.append((maxVal, maxLoc, (int(xt), int(yt))))
    matches.sort(key=lambda e: e[0], reverse=True)
    if len(matches) >= max:
        return np.int32([matches[i][1:] for i in range(max)])
    else:
        return np.int32([c[1:] for c in matches])
