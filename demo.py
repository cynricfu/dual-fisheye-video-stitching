#!/usr/bin/python
import numpy as np
import cv2
import sys
import argparse
import random
import utils


# parameters for template matching
W_max = 1380
offsetYL = 320
offsetYR = 320
maxL = 160
maxR = 160


def main(input, output):
    cap = cv2.VideoCapture(input)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 30.0, (2560, 1280))

    # Obtain xmap and ymap
    xmap, ymap = utils.buildmap_2(
        Ws=W_max, Hs=1280, Wd=1280, Hd=1280, fov=194.0)

    # Calculate homography from the first frame
    ret, frame = cap.read()
    if ret == True:
        # defish / unwarp
        cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
        cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

        # shift the remapped images along x-axis
        shifted_cams = np.zeros((2560, 2560, 3), np.uint8)
        shifted_cams[1280:, (2560 - W_max) / 2:(2560 + W_max) / 2] = cam2
        shifted_cams[:1280, :W_max / 2] = cam1[:, W_max / 2:]
        shifted_cams[:1280, 2560 - W_max / 2:] = cam1[:, :W_max / 2]

        # find matches and extract pairs of correspondent matching points
        matchesL = utils.getMatches_templmatch(
            cam1[offsetYL:1280 - offsetYL, 1280:],
            cam2[offsetYL:1280 - offsetYL, :W_max - 1280],
            (32, 16), maxL)
        matchesR = utils.getMatches_templmatch(
            cam1[offsetYR:1280 - offsetYR, :W_max - 1280],
            cam2[offsetYR:1280 - offsetYR, 1280:],
            (32, 16), maxR)
        print matchesL.shape[0], matchesR.shape[0]

        matchesL = matchesL + ((2560 - W_max) / 2, offsetYL)
        matchesR = matchesR + ((2560 - W_max) / 2 + 1280, offsetYR)
        zipped_matches = zip(matchesL, matchesR)
        matches = np.int32([e for i in zipped_matches for e in i])
        pts1 = matches[:, 0]
        pts2 = matches[:, 1]

        # find homography matrix from pairs of correspondent matching points
        H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 2.0)
        print H

        # warp cam2 using H
        warped2 = cv2.warpPerspective(shifted_cams[1280:], H, (2560, 1280))
        warped1 = np.zeros((1280, 2560, 3), np.uint8)
        warped1[:, :W_max / 2] = cam1[:, W_max / 2:]
        warped1[:, 2560 - W_max / 2:] = cam1[:, :W_max / 2]
        warped = np.zeros((1280, 2560, 3), np.uint8)
        warped[:] = warped2[:]
        warped[:, :W_max / 2] = warped1[:, :W_max / 2]
        warped[:, 2560 - W_max / 2:] = warped1[:, 2560 - W_max / 2:]

        # image labeling (find minimum error boundary cut)
        mask = utils.imgLabeling2(warped1[:, W_max / 2 - 80:W_max / 2],
                                 warped2[:, W_max / 2 - 80:W_max / 2],
                                 warped1[:, 2560 - W_max /
                                         2:2560 - W_max / 2 + 80],
                                 warped2[:, 2560 - W_max /
                                         2:2560 - W_max / 2 + 80],
                                 W_max / 2 - 80, 2560 - W_max / 2)
        labeled = warped1 * mask + warped2 * (1 - mask)

        # multi band blending
        blended = utils.multi_band_blending(warped1, warped2, mask, 6)
        cv2.imshow('p', blended.astype(np.uint8))
        cv2.waitKey(0)

        # write results from phases
        cv2.imwrite('0.png', cam1)
        cv2.imwrite('1.png', cam2)
        cv2.imwrite('2.png', shifted_cams)
        cv2.imwrite('3.png', warped2)
        cv2.imwrite('4.png', warped)
        cv2.imwrite('labeled.png', labeled.astype(np.uint8))
        cv2.imwrite('blended.png', blended.astype(np.uint8))

    # Process each frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # remap
            cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
            cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

            # place the remapped images
            shifted_cams = np.zeros((2560, 2560, 3), np.uint8)
            shifted_cams[1280:, (2560 - W_max) / 2:(2560 + W_max) / 2] = cam2
            shifted_cams[:1280, :W_max / 2] = cam1[:, W_max / 2:]
            shifted_cams[:1280, 2560 - W_max / 2:] = cam1[:, :W_max / 2]

            warped = cv2.warpPerspective(shifted_cams[1280:], H, (2560, 1280))
            warped[:1280, :W_max / 2] = cam1[:, W_max / 2:]
            warped[:1280, 2560 - W_max / 2:] = cam1[:, :W_max / 2]

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
