import cv2
import os
from detector import HG8321R
from detector import HG8321R_load
# from detector import HG260GT
import tensorflow as tf
from net import lenet
import shutil

# imgPath = './data/HG8321R_power1_pon0_los0_lan0/'
# imgName = 'IMG_20171125_102035.jpg'
# img = cv2.imread(imgPath + imgName)
# img = cv2.resize(img, (780, 1052), interpolation=cv2.INTER_CUBIC)
#
# load_model = HG8321R_load.load_model()
# states = HG8321R_load.run_state(img, load_model[0], load_model[1], load_model[2], 'P71125-134101.jpg')
# print(states)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dataFile = './data/unPassTestSrc/'
unPassDataFile = './data/unPassTestDst/'

# dataFile = './data/unPassTestDst/'

picFile = os.listdir(dataFile)

load_model = HG8321R_load.load_model()
for filename in picFile:
    imgPath = dataFile+filename+'/'
    hit = 0

    filename_postfix = os.listdir(imgPath)
    for imgName in filename_postfix:
        try:
            img = cv2.imread(imgPath + imgName)
            img = cv2.resize(img, (780, 1052), interpolation=cv2.INTER_CUBIC)
            states = HG8321R_load.run_state(img, load_model[0], load_model[1], load_model[2], imgName)

            if states == 0:
                print('Fail cluster')
                continue
            if filename.lower() in states["gess_groud_truths"]:
                hit = hit + 1
                # states = HG8321R.stateOutput(img)
                # print(states)
            else:
                shutil.copy(imgPath+imgName, unPassDataFile+filename+'/')
        except:
            print('Error image: {}'.format(imgName))
            continue

    print(filename, "accuracy is :", hit / len(filename_postfix))

# states = HG8321R.stateOutput(img)
# image_size_H = 32
# image_size_W = 32
# num_channels = 3
# X = tf.placeholder(tf.float32, [None, image_size_H, image_size_W, num_channels], name='x-input')
# y_ = tf.argmax(lenet.lenet(X, num_classes=2, is_training=False)[0], 1)
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     states = HG8321R_load.run_state(img, X, y_, sess, 'P71125-134101.jpg')
#     print(states)

