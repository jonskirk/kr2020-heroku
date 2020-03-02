import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

# import matplotlib.pyplot as plt

from train import model


# x = tf.placeholder("float", [None, 784])
x = tf.placeholder("float", [None, 1024])
sess = tf.Session()

# restore trained data
# with tf.variable_scope("regression"):
#     y1, variables = model.regression(x)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/regression.ckpt")


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "train/data/convolutional.ckpt")


# def regression(input):
#     return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# webapp
app = Flask(__name__)
app.cache_size = 0
CORS(app)

# this was our original
# it expects an array of integers from 0 to 255, where 0 = 'on'
@app.route('/api/mnist', methods=['POST'])
def mnist():
    # input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)

    # we pass the stroke number through, in case these get processed in the wrong order due to transit delays
    strokeno = request.json["no"]
    arr = np.array(request.json["stroke"], dtype=np.uint8)

    # image = arr.reshape(32,32)
    # plt.imshow(image)
    # plt.show()

    input = ((255 - arr) / 255.0).reshape(1, 1024)
    # output1 = regression(input)
    output2 = convolutional(input)
    # return jsonify(results=[output1, output2])
    # arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return jsonify(results=[strokeno, output2])

# our current recognizer
# this one is expecting an array of 1s and 0s, comma delimited
# the very first entry is an ID, effectively
@app.route('/kr', methods=['POST'])
def kr():
    strokeno = request.json["no"]
    arr = np.array(request.json["stroke"], dtype=np.uint8)
    input = arr.reshape(1, 1024)
    output2 = convolutional(input)
    return jsonify(results=[strokeno, output2])

@app.route('/')
def main():
    response = render_template('index.html')
    return response


if __name__ == '__main__':
    app.run()
