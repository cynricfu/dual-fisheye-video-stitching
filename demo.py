#!/usr/bin/python
import numpy as np
import cv2
import sys
import argparse
from multi_band_blending import multi_band_blending
from stitch import stitch


def buildmap_pgm(pgm_addr):
    pgm = open(pgm_addr)
    lines = pgm.readlines()
    Wd = int(lines[2].split(' ')[0])
    Hd = int(lines[2].split(' ')[1])
    result_map = np.zeros((Hd, Wd), np.float32)
    for y in range(4, 4 + Hd):
        locs = lines[y].split(' ')
        for x in range(Wd):
            result_map.itemset((y - 4, x), int(locs[x]))
    return result_map


def buildmap(Ws, Hs, Wd, Hd, hfovd=160.0, vfovd=160.0):
    # Build the fisheye mapping
    map_x = np.zeros((Hd, Wd), np.float32)
    map_y = np.zeros((Hd, Wd), np.float32)
    vfov = (vfovd / 180.0) * np.pi
    hfov = (hfovd / 180.0) * np.pi
    vstart = ((180.0 - vfovd) / 180.00) * np.pi / 2.0
    hstart = ((180.0 - hfovd) / 180.00) * np.pi / 2.0
    count = 0
    # need to scale to changed range from our
    # smaller cirlce traced by the fov
    xmax = np.cos(hstart)
    xmin = np.cos(hstart + hfov)
    xscale = xmax - xmin
    xoff = xscale / 2.0
    zmax = np.cos(vstart)
    zmin = np.cos(vfov + vstart)
    zscale = zmax - zmin
    zoff = zscale / 2.0
    # Fill in the map, this is slow but
    # we could probably speed it up
    # since we only calc it once, whatever
    for y in range(0, int(Hd)):
        for x in range(0, int(Wd)):
            count = count + 1
            phi = vstart + (vfov * (float(y) / float(Hd)))
            theta = hstart + (hfov * (float(x) / float(Wd)))
            xp = (np.sin(phi) * np.cos(theta) + xoff) / xscale
            zp = (np.cos(phi) + zoff) / zscale
            xS = Ws - (xp * Ws)
            yS = Hs - (zp * Hs)
            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))

    return map_x, map_y


def rotate(img, theta):
    M = cv2.getRotationMatrix2D((640, 640), theta, 1)
    result = cv2.warpAffine(img, M, (1280, 1280))
    return result


def pad(img, pxs, flags):
    l = img[:, :1280, :]
    r = img[:, 1280:, :]

    l = cv2.copyMakeBorder(l, pxs, pxs, pxs, pxs, flags)
    r = cv2.copyMakeBorder(r, pxs, pxs, pxs, pxs, flags)

    result = np.append(l, r, axis=1)
    result = cv2.resize(result, (2560, 1280))
    return result


def y_align(img, pxs):
    l = img[:, :1280, :]
    r = img[:, 1280:, :]

    region_l = l[1280 - pxs:, :, :]
    region_r = r[:pxs, :, :]
    region_l = cv2.flip(region_l, -1)
    region_r = cv2.flip(region_r, -1)

    l_result = np.append(region_r, l[:1280 - pxs, :, :], axis=0)
    r_result = np.append(r[pxs:, :, :], region_l, axis=0)
    result = np.append(l_result, r_result, axis=1)

    return result


def pivot_smooth(img, shape, wd, flags):
    pivot_m1 = img[:, 1280 - wd:1279 + wd, :]
    pivot_l = img[:, :wd, :]
    pivot_r = img[:, 2560 - wd:, :]
    pivot_m2 = np.append(pivot_r, pivot_l, axis=1)
    if flags:
        pivot_m1 = cv2.GaussianBlur(pivot_m1, shape, 0)
        pivot_m2 = cv2.GaussianBlur(pivot_m2, shape, 0)
    else:
        pivot_m1 = cv2.blur(pivot_m1, shape)
        pivot_m2 = cv2.blur(pivot_m2, shape)

    result = np.copy(img)
    result[:, 1280 - wd:1279 + wd, :] = pivot_m1
    result[:, :wd, :] = pivot_m2[:, wd:, :]
    result[:, 2560 - wd:, :] = pivot_m2[:, :wd, :]
    return result


def pivot_stitch(img, wd):
    # Stitch the area in between
    D = stitch(img[:, 1280 - wd:1280], img[:, 1280:1280 + wd], sigma=15.0)

    # Warp backwards
    pt1 = np.dot(D['H'], [wd, 400, 1])
    pt3 = np.dot(D['H'], [wd, 800, 1])
    pt1 = pt1 / pt1[2]
    pt3 = pt3 / pt3[2]
    src = np.zeros((4, 2), np.float32)
    dst = np.zeros((4, 2), np.float32)
    src[0] = [0, 0]
    src[1] = pt1[:2]
    src[2] = [0, 1280]
    src[3] = pt3[:2]
    dst = np.array(src)
    dst[1] = [2 * wd - 1, 400]
    dst[3] = [2 * wd - 1, 800]

    result = np.copy(img)
    M = cv2.getPerspectiveTransform(src, dst)
    result[:, 1280 - wd:1280 +
           wd] = cv2.warpPerspective(D['res'], M, (2 * wd, 1280))
    result[:, 1280 - wd:1280 + wd] = D['res']
    return result


def main(input, output):
    cap = cv2.VideoCapture(input)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 30.0, (2560, 1280))

    # Obtain xmap and ymap
    xmap = buildmap_pgm(
        './RemapFilter/Samsung Gear 2560x1280/xmap_samsung_gear_2560x1280.pgm')
    ymap = buildmap_pgm(
        './RemapFilter/Samsung Gear 2560x1280/ymap_samsung_gear_2560x1280.pgm')
    #xmap, ymap = buildmap(2560, 1280, 2560, 1280)

    # Perform remap for each frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Fisheye rotation
            frame = np.append(frame[:, :1280, :], rotate(
                frame[:, 1280:, :], -0.5), axis=1)

            # Fisheye padding
            frame = pad(frame, 10, cv2.BORDER_REFLECT_101)

            # Remapping, fisheye -> equirectangular
            frame = cv2.remap(frame, xmap, ymap, cv2.INTER_LINEAR)

            # Vertical alignment
            frame = y_align(frame, 3)

            # Stitching
            #frame = pivot_stitch(frame, 200)

            # Pivot smoothing / blending
            #frame = pivot_smooth(frame, (10, 10), 10, False)
            frame = cv2.resize(multi_band_blending(
                frame[:, :1280, :], frame[:, 1280:, :], overlap_w=20, sigma=2.0), (2560, 1280))
            frame = frame.astype(np.uint8)

            ratio = 1.02
            frame[:, 1280:] = cv2.resize(frame[:, 1280:], (1280, int(
                1280 * ratio)))[int(1280 * (ratio - 1) / 2):int(1280 * (ratio - 1) / 2) + 1280, :]

            # Write the remapped frame
            out.write(frame)

            cv2.imshow('frame', frame)
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
