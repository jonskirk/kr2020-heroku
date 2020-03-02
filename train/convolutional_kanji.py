# this is our main training script
# the CNN definition is in model.py
# the kanji bitmaps are imported via kanji_data.py

import os
import model
# import tensorflow as tf
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets("/tmp/data/", one_hot=True)

import kanji_data as kd
# data = kd.get_data()
# print data

from datetime import datetime

# read our test data
test_data = pd.read_csv('/Users/jonskirk/Downloads/bitmaps_test.csv',header=None).values
print("Test bitmap CSV read")
print(test_data.shape)

test_bitmaps = test_data[:, 1:]
test_kanji_one_hot = kd.convert_to_one_hot(test_data[:, 0])

# model
with tf.variable_scope("convolutional"):
    # x = tf.placeholder(tf.float32, [None, 784])
    x = tf.placeholder(tf.float32, [None, 1024])
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)

# train
# y_ = tf.placeholder(tf.float32, [None, 10])
y_ = tf.placeholder(tf.float32, [None, 403])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # for i in range(20000):
    for i in range(6000):
    # for i in range(5000):
    # for i in range(1200):
    # for i in range(10):
        # batch = data.train.next_batch(50)
        # batch = kd.get_batch(limit=50)
        batch = kd.get_batch(limit=60)
        # print(batch)
        # print(batch[0])
        # print(batch[1])
        # break

        # for j in range(5):
        #     arr = batch[0][j].reshape(32,32)
        #     plt.imshow(arr)
        #     plt.imshow(image)

        # print "Batch: "+str(i)
        # print np.where(batch[1]==1)[1]

        # reverse the one hot encoding for this batch to get a list of the numbers it represents
        # this is horribly inefficient, but don't know the most effective way yet!

        # if True:
        if i % 10 == 0:
        # if i % 100 == 0:
            batch_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={x: test_bitmaps, y_: test_kanji_one_hot, keep_prob: 1.0})
            print("step %d, batch accuracy %g, test set accuracy %g, time %s" % (i, batch_accuracy, test_accuracy, datetime.now()))

        # now train on this batch
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'convolutional.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)
