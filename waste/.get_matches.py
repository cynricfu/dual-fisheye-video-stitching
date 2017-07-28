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
    for m
            warped1[:, W_max / 2 - 50:W_max / 2] -
            warped2[:, W_max / 2 - 50:W_max / 2]), axis=2), axis=1)
        minloc = minloc + W_max / 2 - 50
        mask = np.zeros((1280,2560, 3)), n in rawMatches:
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

            warped1[:, W_max / 2 - 50:W_max / 2] -
            warped2[:, W_max / 2 - 50:W_max / 2]), axis=2), axis=1)
        minloc = minloc + W_max / 2 - 50
        mask = np.zeros((1280,2560, 3))
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
