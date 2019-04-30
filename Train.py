import tensorflow as tf
import numpy as np
import math
import os
import CropImg
from net import lenet

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1001

MODEL_SAVE_PATH = "model/HG8321R_model/"
MODEL_NAME = "HG8321R.ckpt"

def train(X_train, y_train, x_test=None, y_test=None, num_classes=2):
    Xd_num = len(X_train)
    image_size_H = X_train.shape[1]
    image_size_W = X_train.shape[2]
    num_channels = 1
    x_test = np.reshape(x_test, [-1, image_size_H, image_size_W, num_channels])
    X = tf.placeholder(tf.float32, [
        None, image_size_H, image_size_W, num_channels], name='x-input')

    y = tf.placeholder(tf.int64, [None], name='y-input')
    is_training = tf.placeholder(tf.bool)

    logits, _ = lenet.lenet(X, num_classes=num_classes, is_training=is_training)
    global_step = tf.Variable(0, trainable=False)

    #variable_averages = tf.train.ExponentialMovingAverage(Inference.MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, num_classes), logits)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        Xd_num / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    correct_prediction = tf.equal(y, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step]):#, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        for e in range(TRAINING_STEPS):
            # shuffle indicies
            train_indicies = np.arange(Xd_num)
            np.random.shuffle(train_indicies)
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd_num / BATCH_SIZE))-1):
                # generate indicies for the batch
                start_idx = (i * BATCH_SIZE) % Xd_num
                idx = train_indicies[start_idx:start_idx + BATCH_SIZE]

                X_rs = np.reshape(X_train[idx], [BATCH_SIZE, image_size_H, image_size_W, num_channels])
                # create a feed dictionary for this batch
                feed_dict = {X: X_rs,
                             y: y_train[idx],
                             is_training: True}

                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
            
            if e % 10 == 0:
                print("After %d training step(s), loss is %g." % (e, loss_value))
                acc_value = sess.run(accuracy, feed_dict={X: x_test, y: y_test, is_training: False})
                print("accuracy is %g" % (acc_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=e)


def main(argv=None):
    Xd, yd = CropImg.imgProcessing('img/HG8321R/')
    print(Xd.shape, yd.shape)
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    train_X = Xd[train_indicies[:9000]]; train_y = yd[train_indicies[:9000]]
    valid_X = Xd[train_indicies[9000:]]; valid_y = yd[train_indicies[9000:]]
    #test_X = Xd[train_indicies[775:]]; test_y = yd[train_indicies[775:]]
    print(train_X.shape, train_y.shape)
    print(valid_X.shape, valid_y.shape)
    #print(test_X.shape, test_y.shape)
    train(train_X, train_y, valid_X, valid_y, num_classes=2)
    '''
    data = DataProcessing.imgProcessing('img/HG260GT/')
    train_X = np.concatenate((data['positive'], data['negative']))
    train_y = np.concatenate((data['positive_label'], data['negative_label']))
    print(train_X.shape, train_y.shape)
    train(train_X, train_y)
    '''
if __name__ == '__main__':
    tf.app.run()