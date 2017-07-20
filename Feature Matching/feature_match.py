#!/usr/bin/python
import numpy as np
import cv2


def buildmap_pgm(pgm_addr):
    pgm = open(pgm_addr)
    lines = pgm.readlines()
    Wd = int(lines[2].split(' ')[0])
    Hd = int(lines[2].split(' ')[1])
    result_map = np.zeros((Hd, Wd), np.float32)
    for y in range(4, 4 + Hd):
        locs = lines[y].split(' ')
        for x in range(Wd):
            result_map.itemset((y - 4, x), int(locs[x]))
    return result_map


def buildmap(Ws, Hs, Wd, Hd, hfovd=160.0, vfovd=160.0):
    # Build the fisheye mapping
    map_x = np.zeros((Hd, Wd), np.float32)
    map_y = np.zeros((Hd, Wd), np.float32)
    vfov = (vfovd / 180.0) * np.pi
    hfov = (hfovd / 180.0) * np.pi
    vstart = ((180.0 - vfovd) / 180.00) * np.pi / 2.0
    hstart = ((180.0 - hfovd) / 180.00) * np.pi / 2.0
    count = 0
    # need to scale to changed range from our
    # smaller cirlce traced by the fov
    xmax = np.cos(hstart)
    xmin = np.cos(hstart + hfov)
    xscale = xmax - xmin
    xoff = xscale / 2.0
    zmax = np.cos(vstart)
    zmin = np.cos(vfov + vstart)
    zscale = zmax - zmin
    zoff = zscale / 2.0
    # Fill in the map, this is slow but
    # we could probably speed it up
    # since we only calc it once, whatever
    for y in range(0, int(Hd)):
        for x in range(0, int(Wd)):
            count = count + 1
            phi = vstart + (vfov * (float(y) / float(Hd)))
            theta = hstart + (hfov * (float(x) / float(Wd)))
            xp = (np.sin(phi) * np.cos(theta) + xoff) / xscale
            zp = (np.cos(phi) + zoff) / zscale
            xS = Ws - (xp * Ws)
            yS = Hs - (zp * Hs)
            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))

    return map_x, map_y


def BFMatch_ORB(img1, img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return (kp1, kp2, matches)


def BFMatch_SIFT(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return (kp1, kp2, good)


def FlannMatch_SIFT(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    return (kp1, kp2, matches, matchesMask)


if __name__ == "__main__":
    img = cv2.imread('extract.png', cv2.IMREAD_COLOR)
    xmap, ymap = buildmap(Ws=1280, Hs=1280, Wd=1280,
                          Hd=1280, hfovd=160, vfovd=160)
    result_l = cv2.remap(img[:, 0:1280, :], xmap, ymap, cv2.INTER_LINEAR)
    result_r = cv2.remap(img[:, 1280:2560, :], xmap, ymap, cv2.INTER_LINEAR)

    # Brute force matching with ORB
    kp1, kp2, matches = BFMatch_ORB(result_l, result_r)
    img3 = cv2.drawMatches(result_l, kp1, result_r, kp2,
                           matches[:10], outImg=None, flags=2)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)

    # Brute force matching with SIFT and ratio test
    kp1, kp2, good = BFMatch_SIFT(result_l, result_r)
    img3 = cv2.drawMatchesKnn(result_l, kp1, result_r,
                              kp2, good, outImg=None, flags=2)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)

    # FLANN based matching with SIFT and ratio test
    kp1, kp2, matches, matchesMask = FlannMatch_SIFT(result_l, result_r)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(result_l, kp1, result_r,
                              kp2, matches, None, **draw_params)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
