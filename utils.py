#!/usr/bin/python
import numpy as np
import cv2


def get_theta_phi(x_proj, y_proj, W, H, fov):
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * np.pi / H

    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = -np.cos(theta_alt) * np.cos(phi_alt)

    return np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)


def buildmap_1(Ws, Hs, Wd, Hd, fov=193.0):
    fov = fov * np.pi / 180.0
    R_max = np.sin(fov / 2) / (1 + np.cos(fov / 2))

    # cartesian coordinates of the projected (square) image
    ys, xs = np.indices((Hs, Ws), np.float32)
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0

    # spherical coordinates
    theta, phi = get_theta_phi(x_proj, y_proj, Ws, Hs, fov)

    # polar coordinates (of the fisheye image)
    R = np.sin(phi) / (1 - np.cos(phi))

    # cartesian coordinates of the fisheye image
    y_fish = R * np.sin(theta)
    x_fish = R * np.cos(theta)

    ymap = (Hd - y_fish * Hd / R_max) / 2.0
    xmap = (Wd + x_fish * Wd / R_max) / 2.0
    return xmap, ymap
