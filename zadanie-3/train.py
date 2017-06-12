from __future__ import print_function, generators, division

import os

import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


STATE_SIZE = 100
LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")


class Model(object):
    def __init__(self):
        x = tf.placeholder(tf.float32, [None, 28 ^ 2])
        y = tf.placeholder(tf.float32, [None, 10])

        # how reshape x to make it batch x 28 x 28
        x_rows = tf.unstack(tf.reshape(x, shape=[-1, 28, 28]), axis=1)

        start_state = tf.random_normal(shape=[STATE_SIZE, 1], dtype=tf.float32)

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

        state = start_state
        for i, row in enumerate(x_rows):
            state, output = rnn_cell(state, row, reuse_variables=(i > 0))

        self.output = output

        first_input = tf.random_normal(shape=[28, 1])
        second_input = tf.random_normal(shape=[28, 1])
        state, first_output = rnn_cell(start_state, first_input)
        _, second_output = rnn_cell(state, second_input, reuse_variables=True)

        self.f = tf.Print(first_output, [first_output])
        self.s = tf.Print(second_output, [second_output])

        self.init = tf.global_variables_initializer()

    def train(self):
        # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        if not os.path.exists('logs'):
            os.makedirs('logs')

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            sess.run(self.init)
            sess.run([self.f, self.s])


if __name__ == '__main__':
    Model().train()
