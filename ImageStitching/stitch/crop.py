import numpy as np
import cv2


l = cv2.imread('l.png')
r = cv2.imread('r.png')

cv2.imwrite('l_cropped.png', l[:, 1080:])
cv2.imwrite('r_cropped.png', r[:,:200])
