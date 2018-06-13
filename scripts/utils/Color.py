#!/usr/bin/env python

import numpy as np
from matplotlib import cm

class Color(object):
    jet = None

    """
      https://github.com/ros-visualization/rviz/blob/kinetic-devel/src/rviz/default_plugin/point_cloud_transformers.cpp
      :return
        Rainbow color (rgb8) from val in [0., 1.]
    """
    @staticmethod
    def get_rainbow_color(val, min_val=0., diff=255):
        value = 1.0 - (val - min_val)
        # restrict value between 0 and 1
        value = max(value, 0.)
        value = min(value, 1.)

        h = value * 5.0 + 1.0
        i = int(h)
        f = h - i
        # if i is even
        if not i&1:
            f = 1 - f
        n = int((1 - f)*diff)
        bgr = [0]*3
        if i <= 1:
            bgr[2] = n; bgr[1] = 0; bgr[0] = 255
        elif i == 2:
            bgr[2] = 0; bgr[1] = n; bgr[0] = 255
        elif i == 3:
            bgr[2] = 0; bgr[1] = 255; bgr[0] = n
        elif i == 4:
            bgr[2] = n; bgr[1] = 255; bgr[0] = 0
        elif i >= 5:
            bgr[2] = 255; bgr[1] = n; bgr[0] = 0
        # print bgr
        return bgr

    @staticmethod
    def init_jet():
        colormap_int = np.zeros((256, 3), np.uint8)
        colormap_float = np.zeros((256, 3), np.float)

        for i in range(0, 256, 1):
            colormap_float[i, 0] = cm.jet(i)[0]
            colormap_float[i, 1] = cm.jet(i)[1]
            colormap_float[i, 2] = cm.jet(i)[2]

            colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
            colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
            colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))

        return colormap_int

    @staticmethod
    def gray2color(gray_array):
        if Color.jet is None:
            Color.jet = Color.init_jet()

        rows, cols = gray_array.shape
        color_array = np.zeros((rows, cols, 3), np.uint8)

        for i in range(0, rows):
            for j in range(0, cols):
                color_array[i, j] = Color.jet[gray_array[i, j]]

        return color_array

    @staticmethod
    def get_jet_color(val):
        if Color.jet is None:
            Color.jet = Color.init_jet()

        idx = int(val)
        idx = max(idx, 0)
        idx = min(idx, 255)
        color = Color.jet[idx]

        return np.array((np.asscalar(np.uint8(color[0])),
                         np.asscalar(np.uint8(color[1])),
                         np.asscalar(np.uint8(color[2]))))