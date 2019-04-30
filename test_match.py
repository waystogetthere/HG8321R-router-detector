#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt

__version__ = '2017.11.08'
__author__ = 'Felix Chan'


img_path = '/Users/Seraphchan/Desktop/test1.jpg'
mask_path = '/Users/Seraphchan/Desktop/mask.jpg'
MIN_MATCH_COUNT = 5
FLANN_INDEX_KDTREE = 0


def match_sift(mask, img):

    sift = cv2.xfeatures2d_SIFT.create()
    kp_mask, des_mask = sift.detectAndCompute(mask, mask=None)
    kp_img, des_img = sift.detectAndCompute(img, mask=None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_mask, des_img, k=2)
    good = []
    for m, n in matches:
        if m.distance < 2.0 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.reshape(np.float32([kp_mask[m.queryIdx].pt for m in good]), [-1, 1, 2])
        dst_pts = np.reshape(np.float32([kp_img[m.trainIdx].pt for m in good]), [-1, 1, 2])
        mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w = mask.shape
        pts = np.reshape(np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]), [-1, 1, 2])
        dst = cv2.perspectiveTransform(pts, mat)
        cv2.polylines(img, [np.int32(dst)], True, 255, 5, cv2.LINE_AA)
        # cv2.imshow('result', img)
        # cv2.waitKey(3)
        # cv2.destroyAllWindows()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matches_mask = None

    draw_params = dict(matchColor=(0, 255, 0), matchesMask=matches_mask, flags=2)
    img3 = cv2.drawMatches(mask, kp_mask, img, kp_img, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()


if __name__ == '__main__':
    test_image = cv2.imread(img_path)
    mask_image = cv2.imread(mask_path)
    # test_img = cv2.resize(test_image, (780, 1052), interpolation=cv2.INTER_CUBIC)
    # mask_img = cv2.resize(mask_image, (5, 12), interpolation=cv2.INTER_CUBIC)
    test_img = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    mask_img = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    sz = test_img.shape
    crop_img = test_img[int(sz[0] / 3 * 2):sz[0], 0:sz[1]]
    match_sift(mask_img, crop_img)
