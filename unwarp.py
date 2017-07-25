#!/usr/bin/python
import numpy as np
import cv2
import sys
import argparse
import utils


def main(input, output):
    cap = cv2.VideoCapture(input)

    # Obtain xmap and ymap
    xmap, ymap = utils.buildmap_1(Ws=1380, Hs=1280, Wd=1280, Hd=1280, fov=193.0)

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter(output, fourcc, 30.0, (2560, 1280))

    # Perform remap for each frame
    #while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        l = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
        r = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)
        cv2.imwrite('0.png',l)
        cv2.imwrite('1.png',r)
        cv2.imwrite('2.png',np.append(l, r, axis=1))
        # Write the remapped frame
        #out.write(np.append(l, r, axis=1))
        #else:
            #break

    # Release everything if job is finished
    cap.release()
    #out.release()
    #cv2.destroyAllWindows()


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
