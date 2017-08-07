#!/usr/bin/python
import numpy as np
import cv2
import argparse
import utils
import graphcut


# parameters for template matching
W = 1280
H = 640
W_remap = 690
offsetYL = 80
offsetYR = 80
maxL = 80
maxR = 80


def main(input, output):
    cap = cv2.VideoCapture(input)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 30.0, (W, H))

    # Obtain xmap and ymap
    xmap, ymap = utils.buildmap_2(
        Ws=W_remap, Hs=H, Wd=1280, Hd=1280, fov=194.0)

    # Calculate homography and other params from the first frame
    ret, frame = cap.read()
    if ret:
        # defish / unwarp
        cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
        cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)
        cam1_gray = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
        cam2_gray = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)

        # shift the remapped images along x-axis
        shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
        shifted_cams[H:, (W - W_remap) / 2:(W + W_remap) / 2] = cam2
        shifted_cams[:H, :W_remap / 2] = cam1[:, W_remap / 2:]
        shifted_cams[:H, W - W_remap / 2:] = cam1[:, :W_remap / 2]

        # find matches and extract pairs of correspondent matching points
        matchesL = utils.getMatches_goodtemplmatch(
            cam1_gray[offsetYL:H - offsetYL, W / 2:],
            cam2_gray[offsetYL:H - offsetYL, :W_remap - W / 2],
            (16, 8), maxL)
        matchesR = utils.getMatches_goodtemplmatch(
            cam2_gray[offsetYR:H - offsetYR, W / 2:],
            cam1_gray[offsetYR:H - offsetYR, :W_remap - W / 2],
            (16, 8), maxR)
        matchesR = matchesR[:, -1::-1]
        print matchesL.shape[0], matchesR.shape[0]

        matchesL = matchesL + ((W - W_remap) / 2, offsetYL)
        matchesR = matchesR + ((W - W_remap) / 2 + W / 2, offsetYR)
        zipped_matches = zip(matchesL, matchesR)
        matches = np.int32([e for i in zipped_matches for e in i])
        pts1 = matches[:, 0]
        pts2 = matches[:, 1]

        # find homography matrix from pairs of correspondent matching points
        M, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 2.0)
        print M

        # warp cam2 using M
        warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
        warped1 = shifted_cams[:H]

        # calculate vertical boundary of warped image, for later cropping
        top, bottom = utils.verticalBoundary(M, W_remap, W, H)
        print top, bottom

        # crop to get a largest rectangle, and resize to maintain resolution
        warped1 = cv2.resize(warped1[top:bottom], (W, H))
        warped2 = cv2.resize(warped2[top:bottom], (W, H))

        # image labeling (find minimum error boundary cut)
        mask = utils.imgLabeling2(warped1[:, W_remap / 2 - 40:W_remap / 2],
                                  warped2[:, W_remap / 2 - 40:W_remap / 2],
                                  warped1[:, W - W_remap /
                                          2:W - W_remap / 2 + 40],
                                  warped2[:, W - W_remap /
                                          2:W - W_remap / 2 + 40],
                                  (W, H), W_remap / 2 - 40, W - W_remap / 2)
        mask = graphcut.find_graph_cut(
            warped1[:, W_remap / 2 - 40:W_remap / 2],
            warped2[:, W_remap / 2 - 40:W_remap / 2],
            warped1[:, W - W_remap / 2:W - W_remap / 2 + 40],
            warped2[:, W - W_remap / 2:W - W_remap / 2 + 40],
            (W, H), W_remap / 2 - 40, W - W_remap / 2)
        mask, minloc_old = utils.imgLabeling3(
            warped1[:, W_remap / 2 - 40:W_remap / 2],
            warped2[:, W_remap / 2 - 40:W_remap / 2],
            warped1[:, W - W_remap / 2:W - W_remap / 2 + 40],
            warped2[:, W - W_remap / 2:W - W_remap / 2 + 40],
            (W, H), W_remap / 2 - 40, W - W_remap / 2)
        labeled = warped1 * mask + warped2 * (1 - mask)

        # fill empty area of warped1 and warped2, to avoid darkening
        warped1[:, W_remap / 2:W - W_remap /
                2] = warped2[:, W_remap / 2:W - W_remap / 2]
        EAof2 = np.zeros((H, W, 3), np.uint8)
        EAof2[:, (W - W_remap) / 2 + 1:(W + W_remap) / 2 - 1] = 255
        EAof2 = cv2.warpPerspective(EAof2, M, (W, H))
        warped2[EAof2 == 0] = warped1[EAof2 == 0]

        # multi band blending
        blended = utils.multi_band_blending(warped1, warped2, mask, 6)

        cv2.imshow('p', blended.astype(np.uint8))
        cv2.waitKey(0)

        # write results from phases
        cv2.imwrite('0.png', cam1)
        cv2.imwrite('1.png', cam2)
        cv2.imwrite('2.png', shifted_cams)
        cv2.imwrite('3.png', warped2)
        cv2.imwrite('4.png', warped1)
        cv2.imwrite('labeled.png', labeled.astype(np.uint8))
        cv2.imwrite('blended.png', blended.astype(np.uint8))

    # Process each frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # defish / unwarp
            cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
            cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

            # shift the remapped images along x-axis
            shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
            shifted_cams[H:, (W - W_remap) / 2:(W + W_remap) / 2] = cam2
            shifted_cams[:H, :W_remap / 2] = cam1[:, W_remap / 2:]
            shifted_cams[:H, W - W_remap / 2:] = cam1[:, :W_remap / 2]

            # warp cam2 using M
            warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
            warped1 = shifted_cams[:H]

            # crop to get a largest rectangle
            # and resize to maintain resolution
            warped1 = cv2.resize(warped1[top:bottom], (W, H))
            warped2 = cv2.resize(warped2[top:bottom], (W, H))

            # image labeling (find minimum error boundary cut)
            mask = utils.imgLabeling2(warped1[:, W_remap / 2 - 40:W_remap / 2],
                                      warped2[:, W_remap / 2 - 40:W_remap / 2],
                                      warped1[:, W - W_remap /
                                              2:W - W_remap / 2 + 40],
                                      warped2[:, W - W_remap /
                                              2:W - W_remap / 2 + 40],
                                      (W, H), W_remap / 2 - 40,
                                      W - W_remap / 2)
            mask, minloc_old = utils.imgLabeling3(
                warped1[:, W_remap / 2 - 40:W_remap / 2],
                warped2[:, W_remap / 2 - 40:W_remap / 2],
                warped1[:, W - W_remap / 2:W - W_remap / 2 + 40],
                warped2[:, W - W_remap / 2:W - W_remap / 2 + 40],
                (W, H), W_remap / 2 - 40, W - W_remap / 2, minloc_old)
            labeled = warped1 * mask + warped2 * (1 - mask)

            # fill empty area of warped1 and warped2, to avoid darkening
            warped1[:, W_remap / 2:W - W_remap /
                    2] = warped2[:, W_remap / 2:W - W_remap / 2]
            warped2[EAof2 == 0] = warped1[EAof2 == 0]

            # multi band blending
            blended = utils.multi_band_blending(warped1, warped2, mask, 6)

            # Write the remapped frame
            out.write(blended.astype(np.uint8))
            cv2.imshow('warped', blended.astype(np.uint8))
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
        description="A summer research project to seamlessly stitch \
                     dual-fisheye video into 360-degree videos")
    ap.add_argument('input', metavar='INPUT.XYZ',
                    help="path to the input dual fisheye video")
    ap.add_argument('-o', '--output', metavar='OUTPUT.XYZ', required=False,
                    default='output.MP4',
                    help="path to the output equirectangular video")

    args = vars(ap.parse_args())
    main(args['input'], args['output'])
