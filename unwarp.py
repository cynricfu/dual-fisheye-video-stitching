#!/usr/bin/python
import numpy as np
import cv2
import sys
import argparse
import random
import utils


# parameters for template matching
W_max = 1318
offsetYL = 320
offsetYR = 320
maxL = 80
maxR = 80


def main(input, output):
    cap = cv2.VideoCapture(input)

    # Obtain xmap and ymap
    xmap, ymap = utils.buildmap_1(
        Ws=W_max, Hs=1280, Wd=1280, Hd=1280, fov=190.0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 30.0, (2560, 1280))

    # Calculate homography from the first frame
    ret, frame = cap.read()
    if ret == True:
        # remap
        l = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
        r = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

        # place the remapped images
        result = np.zeros((2560, 2560, 3), np.uint8)
        result[1280:, (2560 - W_max) / 2:(2560 + W_max) / 2] = r
        result[:1280, :W_max / 2] = l[:, W_max / 2:]
        result[:1280, 2560 - W_max / 2:] = l[:, :W_max / 2]

        # get matches and extract pairs of correspondent matching points
        matchesL = utils.getMatches_templmatch(
            l[offsetYL:1280 - offsetYL, 1280:], r[offsetYL:1280 - offsetYL, :W_max - 1280], (24, 8), maxL)
        matchesR = utils.getMatches_templmatch(
            l[offsetYR:1280 - offsetYR, :W_max - 1280], r[offsetYR:1280 - offsetYR, 1280:], (12, 6), maxR)

        #matchesL = utils.getMatches_SIFT_FLANN(
        #    l[:, 1280:], r[:, :W_max - 1280], 1, maxL)
        #matchesR = utils.getMatches_SIFT_FLANN(
        #    l[:, :W_max - 1280], r[:, 1280:], 1, maxR)

        print matchesL.shape[0], matchesR.shape[0]
        matchesL = matchesL + ((2560 - W_max) / 2, offsetYL)
        matchesR = matchesR + ((2560 - W_max) / 2 + 1280, offsetYR)
        zipped_matches = zip(matchesL, matchesR)
        matches = np.float32([e for i in zipped_matches for e in i])
        pts1 = matches[:, 0]
        pts2 = matches[:, 1]

        # get warp matrix from pairs of correspondent matching points
        A, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 2.0)
        print A
        warped = cv2.warpPerspective(result[1280:], A, (2560, 1280))
        cv2.imwrite('3.png', warped)
        warped[:1280, :W_max / 2] = l[:, W_max / 2:]
        warped[:1280, 2560 - W_max / 2:] = l[:, :W_max / 2]
        cv2.imwrite('0.png', l)
        cv2.imwrite('1.png', r)
        cv2.imwrite('2.png', result)
        cv2.imwrite('4.png', warped)

    # Perform remap for each frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # remap
            l = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
            r = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

            # place the remapped images
            result = np.zeros((2560, 2560, 3), np.uint8)
            result[1280:, (2560 - W_max) / 2:(2560 + W_max) / 2] = r
            result[:1280, :W_max / 2] = l[:, W_max / 2:]
            result[:1280, 2560 - W_max / 2:] = l[:, :W_max / 2]

            warped = cv2.warpPerspective(result[1280:], A, (2560, 1280))
            warped[:1280, :W_max / 2] = l[:, W_max / 2:]
            warped[:1280, 2560 - W_max / 2:] = l[:, :W_max / 2]

            # Write the remapped frame
            out.write(warped.astype(np.uint8))
            cv2.imshow('warped', warped)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="A summer research project to seamlessly stitch dual-fisheye video into 360-degree videos")
    ap.add_argument('input', metavar='INPUT.XYZ',
                    help="path to the input dual fisheye video")
    ap.add_argument('-o', '--output', metavar='OUTPUT.XYZ', required=False, default='output.MP4',
                    help="path to the output equirectangular video")

    args = vars(ap.parse_args())
    main(args['input'], args['output'])
