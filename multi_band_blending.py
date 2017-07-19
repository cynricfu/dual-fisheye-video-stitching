#!/usr/bin/python
import numpy as np
import cv2
import sys
import argparse


def preprocess(img1, img2, overlap_w, half):
    if img1.shape[0] != img2.shape[0]:
        print "error: image dimension error"
        sys.exit()
    if overlap_w > img1.shape[1] or overlap_w > img2.shape[1]:
        print "error: overlapped area too large"
        sys.exit()

    h = img1.shape[0]
    w1 = img1.shape[1]
    w2 = img2.shape[1]

    if half:
        shape = np.array(img1.shape)
        shape[1] = w1 / 2 + w2 / 2

        subA = np.zeros(shape)
        subA[:, :w1 / 2 + overlap_w / 2] = img1[:, :w1 / 2 + overlap_w / 2]
        subB = np.zeros(shape)
        subB[:, w1 / 2 - overlap_w / 2:] = img2[:,
                                                w2 - (w2 / 2 + overlap_w / 2):]
        mask = np.zeros(shape)
        mask[:, :w1 / 2] = 1
    else:
        shape = np.array(img1.shape)
        shape[1] = w1 + w2 - overlap_w

        subA = np.zeros(shape)
        subA[:, :w1] = img1
        subB = np.zeros(shape)
        subB[:, w1 - overlap_w:] = img2
        mask = np.zeros(shape)
        mask[:, :w1 - overlap_w / 2] = 1

    return subA, subB, mask


def GaussianPyramid(img, sigma, levels):
    GP = [img]
    for i in range(levels - 1):
        GP.append(cv2.resize(cv2.GaussianBlur(
            GP[i], (0, 0), sigmaX=sigma, sigmaY=sigma), (GP[i].shape[1] / 2, GP[i].shape[0] / 2)))
    return GP


def LaplacianPyramid(GP):
    LP = []
    for i, G in enumerate(GP):
        if i == len(GP) - 1:
            LP.append(G)
        else:
            LP.append(G - cv2.resize(GP[i + 1], (G.shape[1], G.shape[0])))
    return LP


def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        True
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended


def reconstruct(LS):
    levels = len(LS)
    R = LS[levels - 1]
    for i in range(levels - 1):
        R = LS[levels - i - 2] + \
            cv2.resize(R, (LS[levels - i - 2].shape[1],
                           LS[levels - i - 2].shape[0]))
    return R


def multi_band_blending(img1, img2, overlap_w, sigma=2.0, levels=None, half=False):
    if overlap_w < 0:
        print "error: overlap_w should be a positive integer"
        sys.exit()
    if sigma <= 0:
        print "error: sigma should be a positive real number"
        sys.exit()
    max_levels = int(np.floor(
        np.log2(min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]))))
    if levels is None:
        levels = max_levels
    if levels < 1 or levels > max_levels:
        print "warning: inappropriate number of levels"
        levels = max_levels

    subA, subB, mask = preprocess(img1, img2, overlap_w, half)

    # Get Gaussian pyramid and Laplacian pyramid
    GPA = GaussianPyramid(subA, sigma, levels)
    GPB = GaussianPyramid(subB, sigma, levels)
    MP = GaussianPyramid(mask, sigma, levels)
    LPA = LaplacianPyramid(GPA)
    LPB = LaplacianPyramid(GPB)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="A Python implementation of multi-band blending")
    ap.add_argument('-H', '--half', required=False, action='store_true',
                    help="option to blend the left half of the first image and the right half of the second image")
    ap.add_argument('-f', '--first', required=True,
                    help="path to the first (left) image")
    ap.add_argument('-s', '--second', required=True,
                    help="path to the second (right) image")
    ap.add_argument('-o', '--overlap', required=True, type=int,
                    help="width of the overlapped area between two images, even number recommended")
    ap.add_argument('-S', '--sigma', required=False, type=float, default=2.0,
                    help="standard deviation of Gaussian function, 2.0 by default")
    ap.add_argument('-l', '--levels', required=False, type=int,
                    help="number of levels of multi-band blending, calculated from image size if not provided")
    args = vars(ap.parse_args())

    half = args['half']
    img1 = cv2.imread(args['first'])
    img2 = cv2.imread(args['second'])
    overlap_w = args['overlap']
    sigma = args['sigma']
    levels = args['levels']

    result = multi_band_blending(img1, img2, overlap_w, sigma, levels, half)
    cv2.imwrite('result.png', result)
