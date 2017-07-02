#!/usr/bin/python
import numpy as np
import cv2


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
    img_l = cv2.imread('l.png', cv2.IMREAD_COLOR)
    img_r = cv2.imread('r.png', cv2.IMREAD_COLOR)

    # Brute force matching with SIFT and ratio test
    kp1, kp2, good = BFMatch_SIFT(img_l, img_r)
    img3 = cv2.drawMatchesKnn(img_l, kp1, img_r, kp2,
                              good, outImg=None, flags=2)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)

    # Further keypoints removal
    thrd = 1120
    cor_x = np.array([[kp1[gd[0].queryIdx].pt[0], kp2[gd[0].trainIdx].pt[0]]
                      for gd in good if kp1[gd[0].queryIdx].pt[0] > thrd and kp2[gd[0].trainIdx].pt[0] < 1280 - thrd])
    cor_y = np.array([[kp1[gd[0].queryIdx].pt[1], kp2[gd[0].trainIdx].pt[1]]
                      for gd in good if kp1[gd[0].queryIdx].pt[0] > thrd and kp2[gd[0].trainIdx].pt[0] < 1280 - thrd])
    cropped_good = [gd for gd in good if kp1[gd[0].queryIdx].pt[0]
                    > thrd and kp2[gd[0].trainIdx].pt[0] < 1280 - thrd]

    print cor_x.shape[0]
    print 'x_range', np.ptp(cor_x, axis=0)
    print 'x_std', np.std(cor_x, axis=0)
    print 'x_median', np.median(cor_x, axis=0)
    print 'x_average', np.average(cor_x, axis=0)
    print

    print 'y_range', np.ptp(cor_y, axis=0)
    print 'y_std', np.std(cor_y, axis=0)
    print 'y_median', np.median(cor_y, axis=0)
    print 'y_average', np.average(cor_y, axis=0)
    print

    x_offset = np.array([1280 - c[0] + c[1] for c in cor_x])
    print 'x_offset_range', np.ptp(x_offset, axis=0)
    print 'x_offset_std', np.std(x_offset, axis=0)
    print 'x_offset_med', np.median(x_offset, axis=0)
    print 'x_offset_avg', np.average(x_offset, axis=0)
    print

    y_offset = np.array([c[1] - c[0] for c in cor_y])
    print 'y_offset_range', np.ptp(y_offset, axis=0)
    print 'y_offset_std', np.std(y_offset, axis=0)
    print 'y_offset_med', np.median(y_offset, axis=0)
    print 'y_offset_avg', np.average(y_offset, axis=0)

    img4 = cv2.drawMatchesKnn(img_l, kp1, img_r, kp2,
                              cropped_good, outImg=None, flags=2)
    cv2.imshow('img4', img4)
    cv2.waitKey(0)
