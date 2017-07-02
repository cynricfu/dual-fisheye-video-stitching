#!/usr/bin/python
import numpy as np
import cv2


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


if __name__ == "__main__":
    cap = cv2.VideoCapture('360_0080.MP4')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.MP4', fourcc, 30.0, (2560, 1280))

    # Obtain xmap and ymap
    xmap = buildmap_pgm(
        './Remap Filter/Samsung Gear 2560x1280/xmap_samsung_gear_2560x1280.pgm')
    ymap = buildmap_pgm(
        './Remap Filter/Samsung Gear 2560x1280/ymap_samsung_gear_2560x1280.pgm')
    #xmap, ymap = buildmap(2560, 1280, 2560, 1280)

    # Perform remap for each frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Padding
            frame = pad(frame, 5, cv2.BORDER_REFLECT_101)

            # Remapping
            frame = cv2.remap(frame, xmap, ymap, cv2.INTER_LINEAR)

            # Vertical alignment
            frame = y_align(frame, 3)

            # Pivot smoothing
            frame = pivot_smooth(frame, (10, 10), 10, False)

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
