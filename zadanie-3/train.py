from __future__ import print_function, generators, division

import argparse
import os
import sys

import datetime
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


HIDDEN_STATE_SIZE = 100
CONCAT_SIZE = HIDDEN_STATE_SIZE + 28
CONVEYOR_SIZE = 100
LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
TRAINING_ITERS = 20000
DISPLAY_STEP = 100
EPOCH_SIZE = 1000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000


class Model(object):
    def __init__(self):
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
                w_initializer = tf.random_normal_initializer(stddev=1 / CONCAT_SIZE)
                w_i = tf.get_variable('w_i', shape=ifg_shape, initializer=w_initializer)
                w_f = tf.get_variable('w_f', shape=ifg_shape, initializer=w_initializer)
                w_o = tf.get_variable('w_o', shape=o_shape, initializer=w_initializer)
                w_g = tf.get_variable('w_g', shape=ifg_shape, initializer=w_initializer)

                tf.summary.histogram('w_i', w_i)

                b_i = tf.get_variable('b_i', shape=[1, CONVEYOR_SIZE], initializer=tf.zeros_initializer())
                b_f = tf.get_variable('b_f', shape=[1, CONVEYOR_SIZE], initializer=tf.zeros_initializer())
                b_o = tf.get_variable('b_o', shape=[1, HIDDEN_STATE_SIZE], initializer=tf.zeros_initializer())
                b_g = tf.get_variable('b_g', shape=[1, CONVEYOR_SIZE], initializer=tf.zeros_initializer())

                tf.summary.histogram('b_i', b_i)

                init_lsmt = tf.variables_initializer([w_i, w_f, w_o, w_g, b_i, b_f, b_o, b_g])

                i = tf.sigmoid(tf.matmul(hx, w_i) + b_i)
                f = tf.sigmoid(tf.matmul(hx, w_f) + b_f)
                o = tf.sigmoid(tf.matmul(hx, w_o) + b_o)
                g = tf.tanh(tf.matmul(hx, w_g) + b_g)

                tf.summary.histogram('i', i)

                new_c = f * c + i * g
                new_h = o * tf.tanh(new_c)

                tf.summary.histogram('c', new_c)
                tf.summary.histogram('h', new_h)

                return new_c, new_h, init_lsmt

        def rnn_net(x, name='lstm_net'):
            with tf.variable_scope(name):
                x_rows = tf.unstack(x, axis=1)
                c = tf.zeros(shape=[BATCH_SIZE, CONVEYOR_SIZE], dtype=tf.float32)
                h = tf.zeros(shape=[BATCH_SIZE, HIDDEN_STATE_SIZE], dtype=tf.float32)
                for i, row in enumerate(x_rows):
                    c, h, init_lstm = lstm(c, h, row, reuse_variables=(i > 0))

                with tf.variable_scope('final_dense'):
                    weights = tf.get_variable('weights', shape=[HIDDEN_STATE_SIZE, 10],
                                              initializer=tf.random_normal_initializer())
                    bias = tf.get_variable('bias', shape=[1, 10], initializer=tf.zeros_initializer())
                    ret = tf.matmul(h, weights) + bias

                # noinspection PyUnboundLocalVariable
                return ret, init_lstm

        pred, init_lstm = rnn_net(images_reshaped)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('loss', self.cost)
        tf.summary.scalar('accuracy', self.accuracy)

        self.init = [tf.global_variables_initializer(), init_lstm]

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
                sess.run(self.optimizer, feed_dict={self.images: batch_x, self.labels: batch_y})
                if i % DISPLAY_STEP == 0:
                    loss, acc, summary = sess.run([self.cost, self.accuracy, self.summary],
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
