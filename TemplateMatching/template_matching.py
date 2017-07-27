import numpy as np
import cv2


def main():
    src = cv2.imread('src.jpg', cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread('tpl.jpg', cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
    result = cv2.normalize(result, dst=None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX, dtype=-1)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    matchLoc = maxLoc
    draw1 = cv2.rectangle(
        src, matchLoc, (matchLoc[0] + tpl.shape[1], matchLoc[1] + tpl.shape[0]), 0, 2, 8, 0)
    draw2 = cv2.rectangle(
        result, matchLoc, (matchLoc[0] + tpl.shape[1], matchLoc[1] + tpl.shape[0]), 0, 2, 8, 0)
    cv2.imshow('draw1', draw1)
    cv2.imshow('draw2', draw2)
    cv2.waitKey(0)
    print src.shape
    print tpl.shape
    print result.shape
    print matchLoc


if __name__ == '__main__':
    main()
