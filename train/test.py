# our main test script
# we want to run a standardized set of bitmaps against each iteration of the model so we can compare results from run to run
# right now we have 2 each for 403 different components

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

import model
import kanji_data as kd
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/jonskirk/Downloads/bitmaps_test.csv')
data = df.values
print("Test bitmap CSV read")
print(data.shape)

# read the pixel data - 32 x 32 = 1024, 0 or 1
X = data[:, 1:]

Y = kd.convert_to_one_hot(data[:, 0])
print(Y)

x = tf.placeholder("float", [None, 1024])
# sess = tf.Session()

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y, variables = model.convolutional(x, keep_prob)
    
# saver = tf.train.Saver(variables)
# saver.restore(sess, "data/convolutional.ckpt")

y_ = tf.placeholder(tf.float32, [None, 403])
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:

	saver = tf.train.Saver(variables)
	saver.restore(sess, "data/convolutional.ckpt")
	# sess.run(tf.global_variables_initializer())
	# sess.run(accuracy, feed_dict={x: X, y_: Y})
	sess.run(y, feed_dict={x: X, keep_prob: 1.0})

	print(X)
	print(X.shape)

	print(Y)
	print(Y.shape)

	print("correct_prediction")
	print(correct_prediction)

	# OK, now let's get a measury of accuracy
	test_accuracy = accuracy.eval(feed_dict={x: X, y_: Y, keep_prob: 1.0})
	print(test_accuracy)