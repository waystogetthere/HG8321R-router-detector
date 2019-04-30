import tensorflow as tf
from net import lenet
import cv2
import numpy as np
import math

MODEL_SAVE_PATH = "./newModel/"
MODEL_NAME = "HG8321R.ckpt"
NUM_LED = 12

def distance(center1, center2):
    x1 = center1[0]; y1 = center1[1]
    x2 = center2[0]; y2 = center2[1]
    dis = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    return dis

def ignoreDuplicate(boxes):
    target_boxes = [boxes[0]]
    for box in boxes[1:]:
        x1, y1, w1, h1 = box
        center1 = (x1+w1/2, y1+h1/2)
        leng = len(target_boxes)
        for i in range(leng):
            x2, y2, w2, h2 = target_boxes[i]
            center2 = (x2+w2/2, y2+h2/2)
            dis = distance(center1, center2)
            if dis < 20.0:
                if  w1*h1 > w2*h2:
                    target_boxes[i] = box
                break
            else:
                if i == leng - 1:
                    target_boxes.append(box)
    return target_boxes

def judgeState(crop_th):
    num_nonzero = np.count_nonzero(crop_th)
    if num_nonzero > crop_th.shape[0] * crop_th.shape[1] / 2:
        return 1
    else:
        return 0
#--------------------------------------
#   Parameters
#       cap: A video object
#   Return
#       states: Each led state(type:dict)
#----------------------------------------
def stateOutput(cap):
    image_size_H = 32
    image_size_W = 32
    num_channels = 3
    X = tf.placeholder(tf.float32, [
        None, image_size_H, image_size_W, num_channels], name='x-input')
    y_ = tf.argmax(lenet.lenet(X, num_classes=2, is_training=False)[0], 1)

    saver = tf.train.Saver()
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    saver.restore(sess, ckpt.model_checkpoint_path)

    states = {}
    led_seq = []
    for i in range(NUM_LED):
        led_seq.append([])

    num_frame = 0
    ret, frame = cap.read()
    shap = frame.shape
    while (ret):
        if num_frame % 10 == 0:
            mser = cv2.MSER_create(_delta=5, _min_area=30, _max_area=1000)
            crop_frame = frame[int(shap[0]/3*2):shap[0], 0:shap[1]]
            gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 215, 255, cv2.THRESH_BINARY)

            # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, (10,10))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, (10, 10), iterations=5)

            regions, boxes = mser.detectRegions(blur)
            target_boxes = ignoreDuplicate(boxes)

            true_boxes = []
            for box in target_boxes:
                x, y, w, h = box
                crop_img = crop_frame[y:y + h, x:x + w]
                crop_img = cv2.resize(crop_img, (32, 32), interpolation=cv2.INTER_CUBIC)
                crop_img = np.reshape(crop_img, [-1, 32, 32, 3])
                y_pre = sess.run(y_, feed_dict={X: crop_img})
                if y_pre == 1:
                    true_boxes.append(box)

            offset = 18
            num_led = len(true_boxes)
            if num_led == NUM_LED:
                true_boxes = sorted(true_boxes, key=lambda led_: led_[0])  # Sort by the x value
                for i in range(num_led):
                    x, y, w, h = true_boxes[i]
                    center = (int(x+ w / 2 ), int(y+ h / 2 ))
                    x_c = center[0]  - 4
                    y_c = center[1] + offset - 4
                    crop_th = th[y_c:y_c + 8, x_c:x_c + 8]
                    led_seq[i].append(judgeState(crop_th))
                    states[i + 1] = float('%.2f' % np.mean(led_seq[i]))  # Mean keep two decimal places
                    cv2.putText(crop_frame, str(states[i + 1]), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    cv2.rectangle(crop_frame, (x_c, y_c), (x_c + 8, y_c + 8), (0, 0, 255), 1)
                """
        
            for box in true_boxes:
                i += 1
                x, y, w, h = box
                #print(box)
                #crop_img = crop_frame[y:y+h,x:x+w]
                #cv2.imwrite('img/HG260GT/'+ str(i) + '.jpg', crop_img)
                cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                #cv2.circle(crop_frame, (int(x+w/2), int(y+h/2-55)), 10, (0, 255, 0), -1)
        """
            cv2.imshow('cap', crop_frame)
            # cv2.imshow('gray', th)
            cv2.waitKey(0)
        ret, frame = cap.read()
        num_frame += 1
    sess.close()
    return states
