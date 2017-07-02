#!/usr/bin/python
import numpy as np


def buildmap_pgm(pgm_addr):
    pgm = open(pgm_addr)
    lines = pgm.readlines()
    Wd = int(lines[2].split(' ')[0])
    Hd = int(lines[2].split(' ')[1])
    result_map = np.zeros((Wd, Hd), np.float32)
    for y in range(4, 4 + Hd):
        locs = lines[y].split(' ')
        for x in range(Wd):
            result_map.itemset((x, y - 4), int(locs[x]))
    return result_map
