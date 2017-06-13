from __future__ import print_function, generators, division

import argparse
import os
import sys

import datetime
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

HIDDEN_STATE_SIZE = 100
# HIDDEN_STATE_SIZE = 28
CONCAT_SIZE = HIDDEN_STATE_SIZE + 28
CONVEYOR_SIZE = 100
# CONVEYOR_SIZE = 28
LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
TRAINING_ITERS = 20000
DISPLAY_STEP = 100
EPOCH_SIZE = 1000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
LAYERS = 2


class LSTM(object):
    def __init__(self, conveyor_size, hidden_state_size, input_size, name='lstm'):
        super(LSTM, self).__init__()

        self.name = name

        concat_size = hidden_state_size + input_size

        with tf.variable_scope(name):
            ifg_shape = [concat_size, conveyor_size]
            o_shape = [concat_size, hidden_state_size]
            w_initializer = tf.random_normal_initializer(stddev=1 / concat_size)
            self.w_i = tf.get_variable('w_i', shape=ifg_shape, initializer=w_initializer)
            self.w_f = tf.get_variable('w_f', shape=ifg_shape, initializer=w_initializer)
            self.w_o = tf.get_variable('w_o', shape=o_shape, initializer=w_initializer)
            self.w_g = tf.get_variable('w_g', shape=ifg_shape, initializer=w_initializer)

            self.b_i = tf.get_variable('b_i', shape=[1, conveyor_size], initializer=tf.zeros_initializer())
            self.b_f = tf.get_variable('b_f', shape=[1, conveyor_size], initializer=tf.zeros_initializer())
            self.b_o = tf.get_variable('b_o', shape=[1, hidden_state_size], initializer=tf.zeros_initializer())
            self.b_g = tf.get_variable('b_g', shape=[1, conveyor_size], initializer=tf.zeros_initializer())

            # self.init_lsmt = tf.variables_initializer([w_i, w_f, w_o, w_g, b_i, b_f, b_o, b_g])
            vars = ['w_i', 'w_f', 'w_o', 'w_g', 'b_i', 'b_f', 'b_o', 'b_g']
            self.init_lsmt = tf.variables_initializer([self.__dict__[k] for k in vars])

    def __call__(self, c, h, x):
        with tf.name_scope('call_%s' % self.name):
            hx = tf.concat([h, x], axis=1)

            i = tf.sigmoid(tf.matmul(hx, self.w_i) + self.b_i)
            f = tf.sigmoid(tf.matmul(hx, self.w_f) + self.b_f)
            o = tf.sigmoid(tf.matmul(hx, self.w_o) + self.b_o)
            g = tf.tanh(tf.matmul(hx, self.w_g) + self.b_g)

            new_c = f * c + i * g
            new_h = o * tf.tanh(new_c)

            # tf.summary.histogram('c', new_c)
            # tf.summary.histogram('h', new_h)

            return new_c, new_h


class Model(object):
    def __init__(self):
        self.images = tf.placeholder(tf.float32, [BATCH_SIZE, 28 ** 2])
        self.labels = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

        images_reshaped = tf.reshape(self.images, shape=[BATCH_SIZE, 28, 28])

        def rnn_net(x, name='lstm_net'):
            with tf.variable_scope(name):
                x_rows = tf.unstack(x, axis=1)

                lstm_first = LSTM(conveyor_size=CONVEYOR_SIZE,
                                  hidden_state_size=HIDDEN_STATE_SIZE,
                                  input_size=28,
                                  name='first_lstm')
                lstm_second = LSTM(conveyor_size=CONVEYOR_SIZE,
                                   hidden_state_size=HIDDEN_STATE_SIZE,
                                   input_size=HIDDEN_STATE_SIZE,
                                   name='second_lstm')

                c0 = tf.zeros(shape=[BATCH_SIZE, CONVEYOR_SIZE], dtype=tf.float32)
                c1 = tf.zeros(shape=[BATCH_SIZE, CONVEYOR_SIZE], dtype=tf.float32)
                h0 = tf.zeros(shape=[BATCH_SIZE, HIDDEN_STATE_SIZE], dtype=tf.float32)
                h1 = tf.zeros(shape=[BATCH_SIZE, HIDDEN_STATE_SIZE], dtype=tf.float32)

                for i, row in enumerate(x_rows):
                    # for i in range(LAYERS):
                    c0, h0 = lstm_first(c0, h0, row)
                    c1, h1 = lstm_second(c1, h1, h0)

                    # c, h, init_lstm = lstm(c, h, row, reuse_variables=(i > 0))

                with tf.variable_scope('final_dense'):
                    weights = tf.get_variable('weights', shape=[HIDDEN_STATE_SIZE, 10],
                                              initializer=tf.random_normal_initializer())
                    bias = tf.get_variable('bias', shape=[1, 10], initializer=tf.zeros_initializer())
                    ret = tf.matmul(h1, weights) + bias

                # noinspection PyUnboundLocalVariable
                return ret, lstm_first.init_lsmt, lstm_second.init_lsmt

        with tf.name_scope('model'):
            pred, init_lstm0, init_lstm1 = rnn_net(images_reshaped)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels))

        # with tf.name_scope('SGD'):
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

        with tf.name_scope('SGD'):
            # Gradient Descent
            optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            grads = tf.gradients(self.loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            self.apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)

        self.init = [tf.global_variables_initializer(), init_lstm0, init_lstm1]

        self.summary = tf.summary.merge_all()

    def train(self, args):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, validation_size=VALIDATION_SIZE)

        if not os.path.exists('logs'):
            os.makedirs('logs')
        if not os.path.exists('save'):
            os.makedirs('save')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

            sess.run(self.init)

            if args.init_from is not None:
                saver.restore(sess, args.init_from)
                print("Model restored.")

            # try:
            for i in range(TRAINING_ITERS):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                sess.run(self.apply_grads, feed_dict={self.images: batch_x, self.labels: batch_y})
                if i % DISPLAY_STEP == 0:
                    loss, acc, summary = sess.run([self.loss, self.accuracy, self.summary],
                                                  # loss, acc = sess.run([self.cost, self.accuracy],
                                                  feed_dict={self.images: batch_x,
                                                             self.labels: batch_y})
                    print("Iter " + str(i) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))
                    writer.add_summary(summary, global_step=i)
                    sys.stdout.flush()

                # validate at the end of every epoch
                if i % EPOCH_SIZE == 0:
                    validation_result = self.validate(mnist.validation, VALIDATION_SIZE)
                    print("Validation accuracy %g" % validation_result)
                    if validation_result > 0.983:
                        break

            # except KeyboardInterrupt:
            print("Optimization Finished!")
            print("Test accuracy %g" % self.validate(mnist.test, TEST_SIZE))

            save_path = saver.save(sess, "save/model.ckpt")
            print("Model saved in file: %s" % save_path)

    def validate(self, dataset, size):
        acc_list = []
        for _ in range(size // BATCH_SIZE):
            batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
            acc = self.accuracy.eval(feed_dict={self.images: batch_x, self.labels: batch_y})
            acc_list.append(acc)
        return np.mean(acc_list)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved 
                        by previous training process""")
    args = parser.parse_args()
    Model().train(args)


if __name__ == '__main__':
    main()
