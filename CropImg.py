
import cv2
import numpy as np
import math
import os

def distance(center1, center2):
    x1 = center1[0]
    y1 = center1[1]
    x2 = center2[0]
    y2 = center2[1]
    dis = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
    return dis


def ignoreDuplicate(boxes):
    target_boxes = [boxes[0]]
    for box in boxes[1:]:
        x1, y1, w1, h1 = box
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        leng = len(target_boxes)
        for i in range(leng):
            x2, y2, w2, h2 = target_boxes[i]
            center2 = (x2 + w2 / 2, y2 + h2 / 2)
            dis = distance(center1, center2)
            if dis < 20.0:
                if w1 * h1 > w2 * h2:
                    target_boxes[i] = box
                break
            else:
                if i == leng - 1:
                    target_boxes.append(box)
    return target_boxes


def cropImg2File(img, filename):
    shap = img.shape
    mser = cv2.MSER_create(_delta=10, _min_area=50, _max_area=1000)
    crop_frame = img[int(shap[0] / 3 * 2):shap[0], 0:shap[1]]
    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.medianBlur(blur, 5)

    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 11, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    th = cv2.erode(th, kernel, iterations=1)
    # th = cv2.dilate(th, kernel, iterations=1)
    # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=3)
    # th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.medianBlur(th, 9)
    regions, boxes = mser.detectRegions(th)
    target_boxes = ignoreDuplicate(boxes)

    i = 1
    for box in target_boxes:
        x, y, w, h = box
        crop_img = crop_frame[y:y + h, x:x + w]
        cv2.imwrite('img/HG8321R/'+ filename + str(i) + '.jpg', crop_img)
        i += 1

def imgProcessing(path):
    filename_postfix = os.listdir(path + 'positive/')
    Xd = []
    yd = []
    for imgname in filename_postfix:
        img = cv2.imread(path + 'positive/' + imgname,0)
        img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
        Xd.append(img)
        yd.append(1)

    filename_postfix = os.listdir(path + 'negative/')
    for imgname in filename_postfix:
        img = cv2.imread(path + 'negative/' + imgname,0)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        Xd.append(img)
        yd.append(0)

    Xd = np.array(Xd)
    yd = np.array(yd)
    return Xd, yd
