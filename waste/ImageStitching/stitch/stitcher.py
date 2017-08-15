import numpy as np
import cv2
from multi_band_blending import multi_band_blending


class Stitcher:
    def __init__(self):
        pass

    def stitch(self, images, ratio=0.75, reprojThresh=2.0, sigma=2.0, levels=None):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageA, imageB) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA,
                                featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M
        subB = cv2.warpPerspective(
            imageB, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
        subA = np.zeros(
            (imageB.shape[0], imageA.shape[1] + imageB.shape[1], 3))
        subA[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

        # apply multi-band blending
        result = multi_band_blending(
            subA, subB, overlap_w=imageA.shape[1] + imageB.shape[1], sigma=sigma, levels=levels)
        result = result.astype(np.uint8)

        # return stitching result, matches visualization and homography matrix
        vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return {'res': result, 'vis': vis, 'H': H}

    def detectAndDescribe(self, image):
        # check to see if we are using OpenCV 3.X
        if int(cv2.__version__[0]) >= 3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary

        # compute the raw matches
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        rawMatches = flann.knnMatch(featuresA, featuresB, k=2)

        # perform Lowe's ratio test to get actual matches
        matches = []
        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < ratio * n.distance:
                # here queryIdx corresponds to kpsA
                # trainIdx corresponds to kpsB
                matches.append((m.trainIdx, m.queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(
                ptsB, ptsA, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        else:
            # otherwise, no homograpy could be computed
            return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
