def temp(input, output):
    img = cv2.imread(input)
    xmap, ymap = utils.buildmap_1(Ws=800, Hs=800, Wd=800, Hd=800, fov=193.0)
    cv2.imwrite(output, cv2.remap(img, xmap,ymap,cv2.INTER_LINEAR))
