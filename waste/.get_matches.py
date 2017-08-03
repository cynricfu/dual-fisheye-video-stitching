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


def getMatches_goodtemplmatch(img1, img2, templ_shape, max):
    if not np.array_equal(img1.shape, img2.shape):
        print "error: inconsistent array dimention", img1.shape, img2.shape
        sys.exit()
    if not (np.all(templ_shape <= img1.shape[:2]) and np.all(templ_shape <= img2.shape[:2])):
        print "error: template shape shall fit img1 and img2"
        sys.exit()

    feature_params = dict(maxCorners=max, qualityLevel=0.01,
                          minDistance=7, blockSize=7)
    kps1 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    kps2 = cv2.goodFeaturesToTrack(img2, mask=None, **feature_params)

    Hs, Ws = img1.shape[:2]
    Ht, Wt = templ_shape
    matches = []
    for [[xt, yt]] in kps1:
        if int(yt) + Ht > Hs or int(xt) + Wt > Ws:
            continue
        result = cv2.matchTemplate(
            img2, img1[int(yt):int(yt) + Ht, int(xt):int(xt) + Wt], cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal > 0.9:
            matches.append((maxVal, (int(xt), int(yt)), maxLoc))
    for [[xt, yt]] in kps2:
        if int(yt) + Ht > Hs or int(xt) + Wt > Ws:
            continue
        result = cv2.matchTemplate(
            img1, img2[int(yt):int(yt) + Ht, int(xt):int(xt) + Wt], cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal > 0.9:
            matches.append((maxVal, maxLoc, (int(xt), int(yt))))
    matches.sort(key=lambda e: e[0], reverse=True)
    if len(matches) >= max:
        return np.int32([matches[i][1:] for i in range(max)])
    else:
        return np.int32([c[1:] for c in matches])


def getMatches_SIFTtemplmatch(img1, img2, templ_shape, max):
    if not np.array_equal(img1.shape, img2.shape):
        print "error: inconsistent array dimention", img1.shape, img2.shape
        sys.exit()
    if not (np.all(templ_shape <= img1.shape[:2]) and np.all(templ_shape <= img2.shape[:2])):
        print "error: template shape shall fit img1 and img2"
        sys.exit()

    # detect and extract features from the image
    sift = cv2.xfeatures2d.SIFT_create()
    kps1, des1 = sift.detectAndCompute(img1, None)
    kps2, des2 = sift.detectAndCompute(img2, None)

    # convert the keypoints from KeyPoint objects to NumPy arrays
    kps1 = [map(int, kp.pt) for kp in kps1]
    kps2 = [map(int, kp.pt) for kp in kps2]

    Hs, Ws = img1.shape[:2]
    Ht, Wt = templ_shape
    matches = []
    for [xt, yt] in kps1:
        if int(yt) + Ht <= Hs and int(xt) + Wt <= Ws:
            result = cv2.matchTemplate(img2, img1[int(yt):int(
                yt) + Ht, int(xt):int(xt) + Wt], cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            if maxVal > 0.9 and cv2.norm((int(xt), int(yt)), maxLoc, cv2.NORM_L2) < 50:
                matches.append((maxVal, (int(xt), int(yt)), maxLoc))

    for [xt, yt] in kps2:
        if int(yt) + Ht <= Hs and int(xt) + Wt <= Ws:
            result = cv2.matchTemplate(
                img1, img2[int(yt):int(yt) + Ht, int(xt):int(xt) + Wt], cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            if maxVal > 0.9 and cv2.norm((int(xt), int(yt)), maxLoc, cv2.NORM_L2) < 50:
                matches.append((maxVal, maxLoc, (int(xt), int(yt))))

    matches.sort(key=lambda e: e[0], reverse=True)
    if len(matches) >= max:
        return np.int32([matches[i][1:] for i in range(max)])
    else:
        return np.int32([c[1:] for c in matches])
