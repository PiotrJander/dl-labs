from __future__ import print_function, generators, division

import os

import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


HIDDEN_STATE_SIZE = 100
CONCAT_SIZE = HIDDEN_STATE_SIZE + 28
CONVEYOR_SIZE = 100
LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
BATCH_SIZE = 50


weight = 'weight'
bias = 'bias'
value = 'value'
ifog = list('ifog')
ifg = list('ifg')
ifo = list('ifo')


class Model(object):
    def __init__(self):
        # self.images = tf.placeholder(tf.float32, [None, 28 ** 2])
        # self.labels = tf.placeholder(tf.float32, [None, 10])
        self.images = tf.placeholder(tf.float32, [BATCH_SIZE, 28 ** 2])
        self.labels = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

        images_reshaped = tf.reshape(self.images, shape=[BATCH_SIZE, 28, 28])

        def lstm(c, h, x, name='lstm', reuse_variables=False):
            with tf.variable_scope(name):
                if reuse_variables:
                    tf.get_variable_scope().reuse_variables()

                hx = tf.concat([h, x], axis=1)

                ifg_shape = [CONCAT_SIZE, CONVEYOR_SIZE]
                o_shape = [CONCAT_SIZE, HIDDEN_STATE_SIZE]
                w_i = tf.get_variable('w_i', shape=ifg_shape, initializer=tf.random_normal_initializer())
                w_f = tf.get_variable('w_f', shape=ifg_shape, initializer=tf.random_normal_initializer())
                w_o = tf.get_variable('w_o', shape=o_shape, initializer=tf.random_normal_initializer())
                w_g = tf.get_variable('w_g', shape=ifg_shape, initializer=tf.random_normal_initializer())

                b_i = tf.get_variable('b_i', shape=[1, CONVEYOR_SIZE], initializer=tf.zeros_initializer())
                b_f = tf.get_variable('b_f', shape=[1, CONVEYOR_SIZE], initializer=tf.zeros_initializer())
                b_o = tf.get_variable('b_o', shape=[1, HIDDEN_STATE_SIZE], initializer=tf.zeros_initializer())
                b_g = tf.get_variable('b_g', shape=[1, CONVEYOR_SIZE], initializer=tf.zeros_initializer())

                init_lsmt = tf.variables_initializer([w_i, w_f, w_o, w_g, b_i, b_f, b_o, b_g])

                i = tf.sigmoid(tf.matmul(hx, w_i) + b_i)
                f = tf.sigmoid(tf.matmul(hx, w_f) + b_f)
                o = tf.sigmoid(tf.matmul(hx, w_o) + b_o)
                g = tf.tanh(tf.matmul(hx, w_g) + b_g)

                new_c = f * c + i * g
                new_h = o * tf.tanh(new_c)

                return new_c, new_h, init_lsmt

        def rnn_net(x, name='rnn_net'):
            with tf.variable_scope(name):
                x_rows = tf.unstack(x, axis=1)
                c = tf.zeros(shape=[BATCH_SIZE, CONVEYOR_SIZE], dtype=tf.float32)
                h = tf.zeros(shape=[BATCH_SIZE, HIDDEN_STATE_SIZE], dtype=tf.float32)
                for i, row in enumerate(x_rows):
                    c, h, init_lstm = lstm(c, h, row, reuse_variables=(i > 0))

                # noinspection PyUnboundLocalVariable
                return h, init_lstm

        run_batch, init_lstm = rnn_net(images_reshaped)

        self.init = tf.global_variables_initializer()
        self.init_lstm = init_lstm
        self.run_batch = run_batch

        # first_input = tf.random_normal(shape=[BATCH_SIZE, 28])
        # second_input = tf.random_normal(shape=[BATCH_SIZE, 28])
        # start_conveyor = tf.random_normal(shape=[BATCH_SIZE, CONVEYOR_SIZE])
        # start_hidden_state = tf.random_normal(shape=[BATCH_SIZE, HIDDEN_STATE_SIZE])
        # c, h, init_lstm = lstm(start_conveyor, start_hidden_state, first_input)
        # _, out, _ = lstm(c, h, second_input, reuse_variables=True)
        #
        # self.f = tf.Print(h, [h])
        # self.s = tf.Print(out, [out])
        # self.init_lstm = init_lstm

    def train(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        if not os.path.exists('logs'):
            os.makedirs('logs')

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            sess.run([self.init, self.init_lstm])
            t = sess.run(self.run_batch, feed_dict={self.images: batch_x})
            print(t)


if __name__ == '__main__':
    Model().train()
