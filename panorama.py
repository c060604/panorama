# -*- coding:utf-8 -*-

import os
import time
import cv2
import imutils
import numpy as np

DIR = './pics'
R_WIDTH = 8400
WIDTH = 8002
HEIGHT = 4001
BLACK_COLOR = 25
RESULT = './result.jpg'
files = os.listdir(DIR)

def stitch(files):
    # 拼接成全景图
    imgs = []
    for file in files:
        imgs.append(cv2.imread(DIR + '/' + file))
    try_use_gpu = False
    stitcher = cv2.createStitcher(try_use_gpu)
    status, pano = stitcher.stitch(imgs)
    if status == 0:
        return pano
    else:
        return None

def crop(img):
    # 裁掉黑边
    width, height = img.shape[1], img.shape[0]
    if width > R_WIDTH:
        img = imutils.resize(img, width=R_WIDTH)
        width, height = img.shape[1], img.shape[0]

    top, bottom = 0, height
    limit = int(height/8)
    top_limit = limit
    bottom_limit = height - limit

    # top
    c = 0
    while c < width:
        r = 0
        while r < top_limit:
            if sum(img[r,c]) < BLACK_COLOR:
                r = r + 1
            else:
                if r > top:
                    top = r
                break
        c = c + 1
    top = top + 1

    # bottom
    c = 0
    while c < width:
        r = height - 1
        while r > bottom_limit:
            if sum(img[r,c]) < BLACK_COLOR:
                r = r - 1
            else:
                if r < bottom:
                    bottom = r
                break
        c = c + 1
    bottom = bottom -1

    # 裁掉上下的黑边
    tmp = img[top:bottom, 0:width]
    width, height = tmp.shape[1], tmp.shape[0]
    limit = int(height/8)
    left, right = 0, width
    left_limit = limit
    right_limit = width - limit

    # left
    r = 0
    while r < height:
        c = 0
        while c < left_limit:
            if sum(tmp[r,c]) < BLACK_COLOR:
                c = c + 1
            else:
                if c > left:
                    left = c
                break
        r = r + 1

    # right
    r = 0
    while r < height:
        c = width - 1
        while c > right_limit:
            if sum(tmp[r,c]) < BLACK_COLOR:
                c = c - 1
            else:
                if c < right:
                    right = c
                break
        r = r + 1

    # 裁掉左右的黑边
    tmp = tmp[0:height, left:right]
    return tmp

def complement_sky(pano):
    tmp = imutils.resize(pano, width=WIDTH)
    rows, cols = tmp.shape[:2]
    border = HEIGHT - rows

    sky = cv2.imread('sky.jpg')
    sky = imutils.resize(sky, width=WIDTH)
    sky_rows = sky.shape[0]
    start = sky_rows - border
    sky = sky[start:sky_rows]

    # 扩展到w:h=2:1
    img = np.vstack((sky[:,:], tmp[:,:]))

    mark = np.zeros((HEIGHT, WIDTH), np.uint8)
    color = (0, 0, 0)
    mark[0:border,0:cols] = 255

    # 用inpaint方法修复
    img = cv2.inpaint(img, mark, 3, cv2.INPAINT_TELEA)

    # 将天空混合
    tmp_sky = img[0:border,:]
    sky = cv2.addWeighted(tmp_sky, 0.7, sky, 0.3, 0.0)
    img = np.vstack((sky[:,:], img[border:,:]))

    # 对边界进行渐入渐出融合
    start = border - 1
    end = start - 100
    blend = 0.01
    for r in range(start, end, -1):
        img[r,:] = tmp[0,:] * (1 - blend) + sky[r,:] * blend
        blend = blend + 0.01

    # 左右各裁掉1像素，避免黑线出现
    rows, cols = img.shape[:2]
    img = img[1:,1:cols-1]

    # 边界边缘再高斯模糊
    tmp = img[0:border+100, :]
    tmp = cv2.GaussianBlur(tmp, (9, 9), 2.5)
    img = np.vstack((tmp[:,:], img[border+100:,:]))

    return img

if __name__ == '__main__':
    start = time.time()
    files = os.listdir(DIR)
    pano = stitch(files)
    if pano is not None:
        pano = crop(pano)
        # pano = complement_sky(pano)
        cv2.imwrite(RESULT, pano)
    else:
        print('error')
    end = time.time()
    print('cost ' + str(end-start))
