def imgLabeling1(img1, img2, img3, img4, xoffsetL, xoffsetR):
    minlocL = np.argmin(np.sum(np.square(img1.astype(
        np.float64) - img2.astype(np.float64)), axis=2), axis=1)
    minlocR = np.argmin(np.sum(np.square(img3.astype(
        np.float64) - img4.astype(np.float64)), axis=2), axis=1)
    minlocL = minlocL + xoffsetL
    minlocR = minlocR + xoffsetR
    mask = np.ones((1280, 2560, 3), np.float64)
    for i in range(1280):
        mask[i, minlocL[i]:minlocR[i]] = 0
        mask[i, minlocL[i]] = 0.5
        mask[i, minlocR[i]] = 0.5
    return mask
s
