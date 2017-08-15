#!/usr/bin/python
import numpy as np
import cv2
import argparse
import dewarp
import feature_matching
import optimal_seamline
import blending
import cropping
import os

# --------------------------------
# output video resolution
W = 2560
H = 1280
# --------------------------------
# field of view, width of de-warped image
FOV = 194.0
W_remap = 1380
# --------------------------------
# params for template matching
templ_shape = (60, 16)
offsetYL = 160
offsetYR = 160
maxL = 80
maxR = 80
# --------------------------------
# params for optimal seamline and multi-band blending
W_lbl = 120
blend_level = 7
# --------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
# --------------------------------


def Hcalc(cap, xmap, ymap):
    """Calculate and return homography for stitching process."""
    Mlist = []
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for frame_no in np.arange(0, frame_count, int(frame_count / 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
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
            matchesL = feature_matching.getMatches_goodtemplmatch(
                cam1_gray[offsetYL:H - offsetYL, W / 2:],
                cam2_gray[offsetYL:H - offsetYL, :W_remap - W / 2],
                templ_shape, maxL)
            matchesR = feature_matching.getMatches_goodtemplmatch(
                cam2_gray[offsetYR:H - offsetYR, W / 2:],
                cam1_gray[offsetYR:H - offsetYR, :W_remap - W / 2],
                templ_shape, maxR)
            matchesR = matchesR[:, -1::-1]

            matchesL = matchesL + ((W - W_remap) / 2, offsetYL)
            matchesR = matchesR + ((W - W_remap) / 2 + W / 2, offsetYR)
            zipped_matches = zip(matchesL, matchesR)
            matches = np.int32([e for i in zipped_matches for e in i])
            pts1 = matches[:, 0]
            pts2 = matches[:, 1]

            # find homography from pairs of correspondent matchings
            M, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
            Mlist.append(M)
    M = np.average(np.array(Mlist), axis=0)
    print M
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return M


def main(input, output):
    cap = cv2.VideoCapture(input)

    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 30.0, (W, H))

    # obtain xmap and ymap
    xmap, ymap = dewarp.buildmap(Ws=W_remap, Hs=H, Wd=1280, Hd=1280, fov=FOV)

    # calculate homography
    M = Hcalc(cap, xmap, ymap)

    # calculate vertical boundary of warped image, for later cropping
    top, bottom = cropping.verticalBoundary(M, W_remap, W, H)

    # estimate empty (invalid) area of warped2
    EAof2 = np.zeros((H, W, 3), np.uint8)
    EAof2[:, (W - W_remap) / 2 + 1:(W + W_remap) / 2 - 1] = 255
    EAof2 = cv2.warpPerspective(EAof2, M, (W, H))

    # process the first frame
    ret, frame = cap.read()
    if ret:
        # de-warp
        cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
        cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

        # shift the remapped images along x-axis
        shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
        shifted_cams[H:, (W - W_remap) / 2:(W + W_remap) / 2] = cam2
        shifted_cams[:H, :W_remap / 2] = cam1[:, W_remap / 2:]
        shifted_cams[:H, W - W_remap / 2:] = cam1[:, :W_remap / 2]

        # warp cam2 using homography M
        warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
        warped1 = shifted_cams[:H]

        # crop to get a largest rectangle, and resize to maintain resolution
        warped1 = cv2.resize(warped1[top:bottom], (W, H))
        warped2 = cv2.resize(warped2[top:bottom], (W, H))

        # image labeling (find minimum error boundary cut)
        mask, minloc_old = optimal_seamline.imgLabeling(
            warped1[:, W_remap / 2 - W_lbl:W_remap / 2],
            warped2[:, W_remap / 2 - W_lbl:W_remap / 2],
            warped1[:, W - W_remap / 2:W - W_remap / 2 + W_lbl],
            warped2[:, W - W_remap / 2:W - W_remap / 2 + W_lbl],
            (W, H), W_remap / 2 - W_lbl, W - W_remap / 2)

        labeled = warped1 * mask + warped2 * (1 - mask)

        # fill empty area of warped1 and warped2, to avoid darkening
        warped1[:, W_remap / 2:W - W_remap /
                2] = warped2[:, W_remap / 2:W - W_remap / 2]
        warped2[EAof2 == 0] = warped1[EAof2 == 0]

        # multi band blending
        blended = blending.multi_band_blending(
            warped1, warped2, mask, blend_level)

        cv2.imshow('p', blended.astype(np.uint8))
        cv2.waitKey(0)

        # write results from phases
        out.write(blended.astype(np.uint8))
        cv2.imwrite(dir_path + '/output/0.png', cam1)
        cv2.imwrite(dir_path + '/output/1.png', cam2)
        cv2.imwrite(dir_path + '/output/2.png', shifted_cams)
        cv2.imwrite(dir_path + '/output/3.png', warped2)
        cv2.imwrite(dir_path + '/output/4.png', warped1)
        cv2.imwrite(dir_path + '/output/labeled.png', labeled.astype(np.uint8))
        cv2.imwrite(dir_path + '/output/blended.png', blended.astype(np.uint8))

    # process each frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # de-warp
            cam1 = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
            cam2 = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)

            # shift the remapped images along x-axis
            shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
            shifted_cams[H:, (W - W_remap) / 2:(W + W_remap) / 2] = cam2
            shifted_cams[:H, :W_remap / 2] = cam1[:, W_remap / 2:]
            shifted_cams[:H, W - W_remap / 2:] = cam1[:, :W_remap / 2]

            # warp cam2 using homography M
            warped2 = cv2.warpPerspective(shifted_cams[H:], M, (W, H))
            warped1 = shifted_cams[:H]

            # crop to get a largest rectangle
            # and resize to maintain resolution
            warped1 = cv2.resize(warped1[top:bottom], (W, H))
            warped2 = cv2.resize(warped2[top:bottom], (W, H))

            # image labeling (find minimum error boundary cut)
            mask, minloc_old = optimal_seamline.imgLabeling(
                warped1[:, W_remap / 2 - W_lbl:W_remap / 2],
                warped2[:, W_remap / 2 - W_lbl:W_remap / 2],
                warped1[:, W - W_remap / 2:W - W_remap / 2 + W_lbl],
                warped2[:, W - W_remap / 2:W - W_remap / 2 + W_lbl],
                (W, H), W_remap / 2 - W_lbl, W - W_remap / 2, minloc_old)

            labeled = warped1 * mask + warped2 * (1 - mask)

            # fill empty area of warped1 and warped2, to avoid darkening
            warped1[:, W_remap / 2:W - W_remap /
                    2] = warped2[:, W_remap / 2:W - W_remap / 2]
            warped2[EAof2 == 0] = warped1[EAof2 == 0]

            # multi band blending
            blended = blending.multi_band_blending(
                warped1, warped2, mask, blend_level)

            # write the remapped frame
            out.write(blended.astype(np.uint8))
            cv2.imshow('warped', blended.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # release everything if job is finished
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
                    default=dir_path + '/output/output.MP4',
                    help="path to the output equirectangular video")

    args = vars(ap.parse_args())
    main(args['input'], args['output'])
