from __future__ import print_function, generators, division

import argparse
import os
import sys

import datetime
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

HIDDEN_STATE_SIZE = 100
CONVEYOR_SIZE = 100
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
            # w_initializer = tf.random_normal_initializer(stddev=1 / concat_size)
            w_initializer = tf.random_normal_initializer(stddev=1 / concat_size)
            self.w_i = tf.get_variable('w_i', shape=ifg_shape, initializer=w_initializer)
            self.w_f = tf.get_variable('w_f', shape=ifg_shape, initializer=w_initializer)
            self.w_o = tf.get_variable('w_o', shape=o_shape, initializer=w_initializer)
            self.w_g = tf.get_variable('w_g', shape=ifg_shape, initializer=w_initializer)

            self.b_i = tf.get_variable('b_i', shape=[1, conveyor_size], initializer=tf.zeros_initializer())
            self.b_f = tf.get_variable('b_f', shape=[1, conveyor_size], initializer=tf.zeros_initializer())
            self.b_o = tf.get_variable('b_o', shape=[1, hidden_state_size], initializer=tf.zeros_initializer())
            self.b_g = tf.get_variable('b_g', shape=[1, conveyor_size], initializer=tf.zeros_initializer())

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
        self.augment = tf.placeholder_with_default(False, shape=[])

        with tf.name_scope('augment'):
            images_no_augmentation = tf.reshape(self.images, shape=[BATCH_SIZE, 28, 28])

            size_gen = lambda: tf.random_uniform(shape=[], minval=24, maxval=29, dtype=tf.int32)
            offset_gen = lambda size: tf.random_uniform(shape=[], minval=0, maxval=28 - size + 1, dtype=tf.int32)

            width = size_gen()
            height = size_gen()

            x_offset = offset_gen(width)
            y_offset = offset_gen(height)

            images = images_no_augmentation[:, x_offset : x_offset + width, y_offset : y_offset + height]

            images = tf.reshape(images, [BATCH_SIZE, -1])
            shape_dim_1 = tf.shape(images)[1]
            # remainder = tf.mod(shape_dim_1, 28)
            # images = tf.pad(images, [[0, 0], [0, 0 if remainder == 0 else 28 - remainder]])
            images = tf.pad(images, [[0, 0], [0, 28 ** 2 - shape_dim_1]])
            images = tf.reshape(images, [BATCH_SIZE, 28, 28])

            tf.summary.image('augmented_image', tf.cast(tf.expand_dims(256 * images[:3], axis=3), dtype=tf.uint8))

            images = tf.cond(self.augment, lambda: images, lambda: images_no_augmentation)

        def rnn_net(x, name='lstm_net'):
            with tf.variable_scope(name):
                x_rows = tf.unstack(x, axis=1)

                def lstm(i, input_size):
                    return LSTM(conveyor_size=CONVEYOR_SIZE,
                                hidden_state_size=HIDDEN_STATE_SIZE,
                                input_size=input_size,
                                name='lstm_%d' % i)

                def deep_lstm(i):
                    return lstm(i, HIDDEN_STATE_SIZE)

                lstms = [lstm(0, 28)] + [deep_lstm(i) for i in range(1, LAYERS)]

                c = tf.zeros(shape=[BATCH_SIZE, CONVEYOR_SIZE], dtype=tf.float32)
                h = tf.zeros(shape=[BATCH_SIZE, HIDDEN_STATE_SIZE], dtype=tf.float32)

                conveyor = [c for _ in range(LAYERS)]
                hidden = [h for _ in range(LAYERS)]

                for i, row in enumerate(x_rows):
                    conveyor[0], hidden[0] = lstms[0](conveyor[0], hidden[0], row)
                    for i in range(1, LAYERS):
                        conveyor[i], hidden[i] = lstms[i](conveyor[i], hidden[i], hidden[i - 1])

                with tf.variable_scope('final_dense'):
                    weights = tf.get_variable('weights', shape=[HIDDEN_STATE_SIZE, 10],
                                              initializer=tf.random_normal_initializer(stddev=1 / HIDDEN_STATE_SIZE))
                    bias = tf.get_variable('bias', shape=[1, 10], initializer=tf.zeros_initializer())
                    ret = tf.matmul(hidden[-1], weights) + bias

                # noinspection PyUnboundLocalVariable
                return ret, [cell.init_lsmt for cell in lstms]

        with tf.name_scope('model'):
            pred, _ = rnn_net(images)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels))

        # with tf.name_scope('SGD'):
        #     self.apply_grads = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

        with tf.name_scope('SGD'):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
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

        self.init = tf.global_variables_initializer()

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
                                                             self.labels: batch_y,
                                                             self.augment: args.augment})
                    print("Iter " + str(i) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))
                    writer.add_summary(summary, global_step=i)
                    sys.stdout.flush()

                # validate at the end of every epoch
                if i % EPOCH_SIZE == 0:
                    validation_result = self.validate(mnist.validation, VALIDATION_SIZE)
                    print("Validation accuracy %g" % validation_result)

                    print('Writer flushed')
                    writer.flush()

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
    parser.add_argument('--augment', type=bool, default=False)
    args = parser.parse_args()
    Model().train(args)


if __name__ == '__main__':
    main()
