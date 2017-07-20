#!/usr/bin/python
from stitcher import Stitcher
import argparse
import cv2


def stitch(img1, img2, sigma, levels, flag_fast):
    if flag_fast:
        if img1.shape[1] > 400:
            img1 = cv2.resize(
                img1, (width, int(img1.shape[0] * float(width) / img1.shape[1])))
        if img2.shape[1] > 400:
            img2 = cv2.resize(
                img2, (width, int(img2.shape[0] * float(width) / img2.shape[1])))

    # stitch the images together to create a panorama
    stitcher = Stitcher()
    return stitcher.stitch([img1, img2], sigma=sigma, levels=levels,
                           showMatches=True)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="A Python script for imgage stitching (panorama generating)")
    ap.add_argument('-F', '--fast', required=False, action='store_true',
                    help="option to resize images (to of width 400) for faster processing")
    ap.add_argument('-f', '--first', required=True,
                    help="path to the first image")
    ap.add_argument('-s', '--second', required=True,
                    help="path to the second image")
    ap.add_argument('-S', '--sigma', required=False, type=float, default=2.0,
                    help="standard deviation of Gaussian function, 2.0 by default")
    ap.add_argument('-l', '--levels', required=False, type=int,
                    help="number of levels of multi-band blending, calculated from image size if not provided")
    args = vars(ap.parse_args())

    # load the two images and other arguments
    flag_fast = args['fast']
    img1 = cv2.imread(args['first'])
    img2 = cv2.imread(args['second'])
    sigma = args['sigma']
    levels = args['levels']

    # stitch the two images
    resvis = stitch(img1, img2, sigma, levels, flag_fast)

    if resvis is None:
        print "there aren't enough matched keypoints to create a panorama"
    else:
        # show the images
        result, vis = resvis
        cv2.imshow("Image A", img1)
        cv2.imshow("Image B", img2)
        cv2.imshow("Keypoint Matches", vis)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
