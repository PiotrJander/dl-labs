from __future__ import print_function, generators, division

import os

import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


STATE_SIZE = 100
LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
BATCH_SIZE = 50


class Model(object):
    def __init__(self):
        # self.images = tf.placeholder(tf.float32, [None, 28 ** 2])
        # self.labels = tf.placeholder(tf.float32, [None, 10])
        self.images = tf.placeholder(tf.float32, [BATCH_SIZE, 28 ** 2])
        self.labels = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

        images_reshaped = tf.reshape(self.images, shape=[-1, 28, 28])
        images_reshaped = tf.expand_dims(images_reshaped, axis=3)

        def rnn_cell(h, x, name='rnn', reuse_variables=False):
            with tf.variable_scope(name):
                if reuse_variables:
                    tf.get_variable_scope().reuse_variables()

                w_hh = tf.get_variable('w_hh',
                                       shape=[STATE_SIZE, STATE_SIZE],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
                w_xh = tf.get_variable('w_xh',
                                       shape=[STATE_SIZE, 28],
                                       dtype=tf.float32,
                                       initializer=tf.random_normal_initializer())
                w_hy = tf.get_variable('w_hy',
                                       shape=[10, STATE_SIZE],
                                       dtype=tf.float32,
                                       initializer=tf.random_normal_initializer())

                # update hidden state
                new_h = tf.tanh(tf.matmul(w_hh, h) + tf.matmul(w_xh, x))

                # compute the output vector
                y = tf.matmul(w_hy, new_h)
                return new_h, y

        def rnn_net(x, name='rnn_net', reuse_variables=False):
            with tf.variable_scope(name):
                if reuse_variables:
                    tf.get_variable_scope().reuse_variables()

                x_rows = tf.unstack(x, axis=1)
                h = tf.random_normal(shape=[STATE_SIZE, 1], dtype=tf.float32)
                for i, row in enumerate(x_rows):
                    h, y = rnn_cell(h, row, reuse_variables=(i > 0))

                # noinspection PyUnboundLocalVariable
                return y

        # self.run_batch = tf.map_fn(rnn_net, images_reshaped)
        # self.run_batch = rnn_net(images_reshaped[0])
        self.run_batch = rnn_net(images_reshaped)
        # self.run_batch = tf.stack([
        #     rnn_net(x, reuse_variables=(i > 0)) for i, x in enumerate(tf.unstack(images_reshaped))])

        self.init = tf.global_variables_initializer()

        # first_input = tf.random_normal(shape=[28, 1])
        # second_input = tf.random_normal(shape=[28, 1])
        # state, first_output = rnn_cell(start_state, first_input)
        # _, second_output = rnn_cell(state, second_input, reuse_variables=True)
        #
        # self.f = tf.Print(first_output, [first_output])
        # self.s = tf.Print(second_output, [second_output])

    def train(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        if not os.path.exists('logs'):
            os.makedirs('logs')

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            sess.run(self.init)
            sess.run(self.run_batch, feed_dict={self.images: batch_x})


if __name__ == '__main__':
    Model().train()
