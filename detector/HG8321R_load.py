# -*- coding: utf-8 -*-
import tensorflow as tf
from net import lenet
import cv2
import numpy as np
import math
import base64
import os.path

# MODEL_SAVE_PATH = "./modelcxc"
# MODEL_NAME = "HG8321R1211.ckpt"
MODEL_SAVE_PATH = "./modelhk"
MODEL_NAME = "HG8321R.ckpt"
NUM_LED = 6

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


def judgeState(crop_th):
    num_nonzero = np.count_nonzero(crop_th)
    if num_nonzero != 0:
        return 1
    else:
        return 0

# -------------------------------dbscan-------------------------
UNCLASSIFIED = False
NOISE = 0

def dist(a, b):
    """
    输入：向量A, 向量B
    输出：两个向量的欧式距离
    """
    return math.sqrt(np.power(a - b, 2).sum())

def eps_neighbor(a, b, eps):
    """
    输入：向量A, 向量B
    输出：是否在eps范围内
    """
    return dist(a, b) < eps

def region_query(data, pointId, eps):
    """
    输入：数据集, 查询点id, 半径大小
    输出：在eps范围内的点的id
    """
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    """
    输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
    输出：能否成功分类
    """
    seeds = region_query(data, pointId, eps)
    if len(seeds) < minPts:  # 不满足minPts条件的为噪声点
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId  # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0:  # 持续扩张
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True

def dbscan(data, eps, minPts):
    """
    输入：数据集, 半径大小, 最小点个数
    输出：分类簇id
    """
    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        point = data[:, pointId]
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1
# -------------------------------------------------------------------------------------


# --------------------------------------
#   Parameters
#       cap: A img object
#   Return
#       states: Each led state(type:dict)
# ----------------------------------------
def load_model():
    image_size_H = 32
    image_size_W = 32
    num_channels = 1

    X = tf.placeholder(tf.float32, [
        None, image_size_H, image_size_W, num_channels], name='x-input')
    y_ = tf.argmax(lenet.lenet(X, num_classes=2, is_training=False)[0], 1)

    saver = tf.train.Saver()
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    saver.restore(sess, ckpt.model_checkpoint_path)
    return X,y_,sess


def interface_chg(states,image_name):
  image_names=[]
  image_names.append(image_name) 
  gess_groud_truths=[]
  if states=={}:
    result_str='g140wc_power0_pon0_los0_lan0'
    gess_groud_truths.append(result_str)
  elif states=={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}:
    result_str='hg8321r_power0_pon0_los0_lan0'
    gess_groud_truths.append(result_str)
  elif states=={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}:
    result_str='hg8321r_power1_pon0_los0_lan0'
    gess_groud_truths.append(result_str)
  elif states=={1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}:
    result_str='hg8321r_power1_pon1_los0_lan0'
    gess_groud_truths.append(result_str)
  elif states=={1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0}:
    result_str='hg8321r_power1_pon1_los0_lan1'
    gess_groud_truths.append(result_str)
  elif states=={1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0}:
    result_str='hg8321r_power1_pon0_los1_lan0'
    gess_groud_truths.append(result_str)
  elif states=={1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0}:
    result_str='hg8321r_power1_pon0_los1_lan1'
    gess_groud_truths.append(result_str)
  elif states=={1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0}:
    result_str='hg8321r_power1_pon1_los0_lan1'
    gess_groud_truths.append(result_str)
  elif states=={1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0}:
      result_str = 'hg8321r_power1_pon0_los0_lan1'
      gess_groud_truths.append(result_str)
  else:
    result_str='g140wc_power1_pon1_los1_lan1'
    gess_groud_truths.append(result_str)
  # print(states)
  result = {'image_names':image_names,'gess_groud_truths':gess_groud_truths}
  # print(result)
  return result




def run_state(img,X,y_,sess,image_name):
    states = {}
    led_seq = []
    for i in range(NUM_LED):
        led_seq.append([])

    shap = img.shape
    mser = cv2.MSER_create(_delta=10, _min_area=50, _max_area=1000)
    crop_frame = img[int(shap[0] / 3 * 0.5):shap[0], 0:shap[1]]
    # crop_frame_b, crop_frame_g, crop_frame_r = crop_frame[:, :, 0], crop_frame[:, :, 1], crop_frame[:, :, 2]
    # crop_frame_new_r = cv2.subtract(crop_frame_r, crop_frame_g)
    # crop_frame_new_g = cv2.subtract(crop_frame_g, crop_frame_r)
    # _, th_binary_new_1 = cv2.threshold(crop_frame_new_r, 30, 255, cv2.THRESH_BINARY)
    # _, th_binary_new_2 = cv2.threshold(crop_frame_new_g, 20, 255, cv2.THRESH_BINARY)
    # th_binary_1 = cv2.bitwise_or(th_binary_new_1, th_binary_new_2)
    # crop_frame_new_y = cv2.subtract(crop_frame_b, crop_frame_g)
    # crop_frame_new_m = cv2.subtract(crop_frame_g, crop_frame_b)
    # _, th_binary_new_3 = cv2.threshold(crop_frame_new_y, 20, 255, cv2.THRESH_BINARY)
    # _, th_binary_new_4 = cv2.threshold(crop_frame_new_m, 20, 255, cv2.THRESH_BINARY)
    # th_binary_2 = cv2.bitwise_or(th_binary_new_3, th_binary_new_4)
    # th_binary_new = cv2.bitwise_or(th_binary_1, th_binary_2)

    # 转换到 HSV
    hsv = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV)
    # 设定绿色的阈值
    lower_g = np.array([30, 40, 120])
    upper_g = np.array([77, 255, 255])
    # 设定红色的阈值
    lower_r = np.array([156, 20, 120])
    upper_r = np.array([180, 255, 255])
    # 根据阈值构建掩模
    mask_g = cv2.inRange(hsv, lower_g, upper_g)
    mask_r = cv2.inRange(hsv, lower_r, upper_r)
    th_binary_new = cv2.bitwise_or(mask_g, mask_r)

    # cv2.imshow('sub', th_binary_new)
    # cv2.imshow('y_', crop_frame_y)
    # cv2.imshow('m_', crop_frame_m)
    # cv2.waitKey(0)
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

    # mser 定位结果
    # for i in range(len(target_boxes)):
    #     x, y, w, h = target_boxes[i]
    #     center = (int(x + w / 2), int(y + h / 2))
    #     x_c = center[0] - 8
    #     y_c = center[1] - 25 - 8
    #     # cv2.putText(crop_frame, str(states[i + 1]), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #
    #     cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #     cv2.rectangle(crop_frame, (x_c, y_c), (x_c + 16, y_c + 16), (0, 0, 255), 1)
    # cv2.imshow("mser detect result", crop_frame)
    # cv2.waitKey(0)


    # 深度学习算法过滤
    true_boxes = []
    data = []
    angleList = []

    for box in target_boxes:
        x, y, w, h = box
        crop_img = gray[y:y + h, x:x + w]
        # cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 1)


        # 对图片进行旋转
        crop_img = cv2.bitwise_not(crop_img)
        th = crop_img.mean() - 5
        thresh = cv2.threshold(crop_img, th, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 17:
            angle = 0
        # else:
        #     print("angle", angle)
        (h, w) = crop_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        crop_img = cv2.warpAffine(crop_img, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        index = []
        mean = crop_img.mean()
        for i in range(len(crop_img)):
            a = crop_img[i].max()
            if a > mean:
                index.append(i)

        if len(index) < len(crop_img) * 0.65:
            crop_img = crop_img[index]
        crop_img = cv2.bitwise_not(crop_img)

        # cv2.imshow("crop_img", crop_img)
        #
        # cv2.waitKey(0)

        crop_img = cv2.resize(crop_img, (32, 32), interpolation=cv2.INTER_CUBIC)
        crop_img = np.reshape(crop_img, [-1, 32, 32, 1])
        y_pre = sess.run(y_, feed_dict={X: crop_img})
        if y_pre == 1:
            true_boxes.append(box)
            data.append([box[0], box[1]])
            angleList.append(angle)

    # 深度学习过滤算法结果
    # for k in range(len(true_boxes)):
    #     x, y, w, h = true_boxes[k]
    #     center = (int(x + w / 2), int(y + h / 2))
    #     x_c = center[0] - 8
    #     y_c = center[1] - 25 - 8
    #     # cv2.putText(crop_frame, str(states[i + 1]), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #
    #     cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #     cv2.rectangle(crop_frame, (x_c, y_c), (x_c + 16, y_c + 16), (0, 0, 255), 1)
    # cv2.imshow("deeplearning result", crop_frame)
    # cv2.waitKey(0)

    # dbscan聚类算法进一步过滤
    data = np.mat(data).transpose()
    clusters, clusterNum = dbscan(data, 100, 3)  # dbscan聚类
    # 求类别数量等于NUM_LED的类别ID
    clusters_key = []
    for i in range(clusterNum + 1):
        clusters_key.append(0)
    for value in clusters:
        clusters_key[value] += 1
    num_led_index = np.where(np.array(clusters_key) == NUM_LED)
    index = np.where(np.array(clusters) == num_led_index[0])
    true_boxes = np.array(true_boxes)[index]
    angleArray = np.array(angleList)[index]

    if len(true_boxes) == 0:
        states = interface_chg(states, image_name)
        # print(states)
        return states

    # Dynamic threshold
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 直方图均衡化
    #equ = clahe.apply(blur)
    #x_dy, y_dy, w_dy, h_dy = true_boxes[0]
    #x_dy = int(x_dy + w_dy / 2) - 4
    #y_dy = int(y_dy + h_dy / 2) + 25 - 4
    #crop_dy = blur[y_dy:y_dy + 8, x_dy:x_dy + 8]
    #th_mean = np.mean(crop_dy)
    #_, th_binary = cv2.threshold(equ, int(th_mean) + 20, 255, cv2.THRESH_BINARY)

    # # 聚类结果
    # for i in range(len(true_boxes)):
    #     x, y, w, h = true_boxes[i]
    #     center = (int(x + w / 2), int(y + h / 2))
    #     x_c = center[0] - 8
    #     y_c = center[1] - 25 - 8
    #     # cv2.putText(crop_frame, str(states[i + 1]), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #     cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #     cv2.rectangle(crop_frame, (x_c, y_c), (x_c + 16, y_c + 16), (0, 0, 255), 1)
    # cv2.imshow("clusting result", crop_frame)
    # cv2.waitKey(0)

    offset = -25
    a = angleArray.mean()
    # print(a)
    num_led = len(true_boxes)
    if num_led == NUM_LED:
        true_boxes = sorted(true_boxes, key=lambda led_: led_[0])  # Sort by the x value
        for i in range(num_led):
            x, y, w, h = true_boxes[i]
            center = (int(x + w / 2), int(y + h / 2))
            # x_c = center[0] - 8
            # y_c = center[1] + offset - 8
            # print("angle:", a)
            x_c = int(center[0] - offset * math.sin(a * math.pi/180) - 8)
            y_c = int(center[1] + offset * math.cos(a * math.pi/180) - 8)
            # print(x_c, y_c, offset * math.sin(a), offset * math.cos(a), math.sin(a), math.cos(a))
            crop_th = th_binary_new[y_c:y_c + 16, x_c:x_c + 16]
            led_seq[i].append(judgeState(crop_th))
            states[i + 1] = float('%.2f' % np.mean(led_seq[i]))  # Mean keep two decimal places
            #cv2.putText(crop_frame, str(states[i + 1]), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            # cv2.rectangle(crop_frame, (x_c, y_c), (x_c + 16, y_c + 16), (0, 0, 255), 1)

    else:
        print("detection of failure!")
    states = interface_chg(states,image_name)
    # cv2.imshow('cap', crop_frame)
    # cv2.imshow('gray', gray)
    # cv2.imshow('th', th_binary)
    # cv2.imshow('th1', th)
    # cv2.imshow('equ', equ)
    # cv2.waitKey(0)
    # print(states)
    return states




def savenewonuimg(files,usr_nbr,filenames,savepath):
  #timesstr = time.localtime()
  # usr_nbr as dirname
  fileslist=files.split(',')
  filenameslist=filenames.split(',')
  dirname = str(usr_nbr)
  path=savepath+'/'+dirname
  for file,filename in zip(fileslist,filenameslist):
    strfile = base64.b64decode(file)
    if os.path.exists(path)== False:
      os.mkdir(path)
      if os.path.exists(path+'/'+filename) == True:
        os.remove(path+'/'+filename)
      imgfile = open(path+'/'+filename, 'wb')
      imgfile.write(strfile)
      imgfile.close()
    elif os.path.exists(path)== True:
      if os.path.exists(path+'/'+filename) == True: 
        os.remove(path+'/'+filename) 
      imgfile = open(path+'/'+filename, 'wb')
      imgfile.write(strfile)
      imgfile.close()
  image=cv2.imread(path+'/'+filename)
  print(path+'/'+filename)
  res_image=cv2.resize(image,(780,1040),interpolation=cv2.INTER_CUBIC)
  return res_image



