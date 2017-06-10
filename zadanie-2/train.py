from __future__ import print_function, generators, division

import datetime
import random
from itertools import count
import os
import sys

import tensorflow as tf
import numpy as np


LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
DATA_SET_SIZE = int(os.environ.get('DATA_SET_SIZE') or 10593)
VALIDATION_SET_SIZE = int(os.environ.get('VALIDATION_SET_SIZE') or 593)
TRAIN_SET_SIZE = DATA_SET_SIZE - VALIDATION_SET_SIZE
DATA_DIR = os.environ.get('SPACENET') or '/data/spacenet2/'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
HEATMAPS_DIR = os.path.join(DATA_DIR, 'heatmaps')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE') or 20)
AUGMENTED_BATCH_SIZE = 8 * BATCH_SIZE
HALF_IMAGE_SIZE = 325
IMAGE_SIZE = 256
CHANNELS = 3
LEARNING_RATE = 1e-3
IMAGE_TRANSFORMATION_NUMBER = 8
num_preprocess_threads = 2
min_queue_examples = 64


class ImagesHeatmaps(object):
    def __init__(self, images=None, heatmaps=None):
        super(ImagesHeatmaps, self).__init__()

        self.images = images if images is not None else {}
        self.heatmaps = heatmaps if heatmaps is not None else {}

    def map(self, f):
        return ImagesHeatmaps(f(self.images), f(self.heatmaps))


class TrainValidate(object):
    def __init__(self, train=None, validate=None):
        super(TrainValidate, self).__init__()

        self.train = train or ImagesHeatmaps()
        self.validate = validate or ImagesHeatmaps()

    def map(self, f):
        return TrainValidate(self.train.map(f), self.validate.map(f))


def conv_relu(features, filters):
    return tf.layers.conv2d(
        features,
        filters=filters,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )


def bn_conv_relu(features, filters):
    return conv_relu(
        tf.layers.batch_normalization(features),
        filters
    )


def max_pool(features):
    return tf.layers.max_pooling2d(
        features,
        pool_size=2,
        strides=2,
        padding='same'
    )


def bn_conv_relu_3_maxpool(features, channels):
    out = bn_conv_relu(features, channels)
    out_skip_conn = bn_conv_relu(out, channels)
    out = bn_conv_relu(out_skip_conn, channels)
    out = max_pool(out)
    return out_skip_conn, out


def upconv_relu(features, filters):
    return tf.layers.conv2d_transpose(
        features,
        filters=filters,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )


def bn_upconv_relu(features, filters):
    return upconv_relu(tf.layers.batch_normalization(features), filters)


def concat_bn_conv_relu_2(features_down, features_up):
    out = tf.concat([features_down, features_up], axis=3)
    out = bn_conv_relu(out, 96)
    return bn_conv_relu(out, 64)


def concat_bn_conv_relu_2_bn_upconv_relu(features_down, features_up):
    out = concat_bn_conv_relu_2(features_down, features_up)
    return bn_upconv_relu(out, 64)


def convout(features):
    return tf.layers.conv2d(
        features,
        filters=3,
        kernel_size=1,
        strides=1
    )


def conv_net(features):
    features_down1 = conv_relu(features, 64)

    features_down2_skip_conn = bn_conv_relu(features_down1, 64)
    features_down2 = bn_conv_relu(features_down2_skip_conn, 64)
    features_down2 = max_pool(features_down2)

    features_down3_skip_conn, features_down3 = bn_conv_relu_3_maxpool(features_down2, 64)
    features_down4_skip_conn, features_down4 = bn_conv_relu_3_maxpool(features_down3, 64)
    features_down5_skip_conn, features_down5 = bn_conv_relu_3_maxpool(features_down4, 64)
    features_down6_skip_conn, features_down6 = bn_conv_relu_3_maxpool(features_down5, 64)

    features_up1 = bn_conv_relu(features_down6, 64)
    features_up1 = bn_conv_relu(features_up1, 64)
    features_up1 = bn_upconv_relu(features_up1, 64)

    features_up2 = concat_bn_conv_relu_2_bn_upconv_relu(features_down6_skip_conn, features_up1)
    features_up3 = concat_bn_conv_relu_2_bn_upconv_relu(features_down5_skip_conn, features_up2)
    features_up4 = concat_bn_conv_relu_2_bn_upconv_relu(features_down4_skip_conn, features_up3)
    features_up5 = concat_bn_conv_relu_2_bn_upconv_relu(features_down3_skip_conn, features_up4)

    features_up6 = concat_bn_conv_relu_2(features_down2_skip_conn, features_up5)
    features_up6 = convout(features_up6)

    for k, t in locals().items():
        tf.summary.histogram(k, t)

    return features_up6


def augment_image(image):
    r = tf.image.rot90
    s = tf.image.flip_left_right
    functions = [
        lambda i: i,
        lambda i: r(i),
        lambda i: r(r(i)),
        lambda i: r(r(r(i))),
        lambda i: s(i),
        lambda i: r(s(i)),
        lambda i: r(r(s(i))),
        lambda i: r(r(r(s(i))))
    ]
    augmented_images = map(lambda f: f(image), functions)
    return tf.stack(augmented_images)


def revert_transformations(augmented_images):
    r = tf.image.rot90
    s = tf.image.flip_left_right
    revert_functions = [
        lambda i: i,
        lambda i: r(r(r(i))),
        lambda i: r(r(i)),
        lambda i: r(i),
        lambda i: s(i),
        lambda i: s(r(r(r(i)))),
        lambda i: s(r(r(i))),
        lambda i: s(r(i))
    ]
    return tf.stack([f(i) for f, i in zip(revert_functions, tf.unstack(augmented_images))])


def gather_transformations(augmented_images):
    return tf.expand_dims(tf.reduce_mean(revert_transformations(augmented_images), 0), 0)


def augment_many(images):
    augmented_batch = tf.map_fn(augment_image, images)
    return tf.reshape(augmented_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])


def get_file_names():
    images_filenames = os.listdir(IMAGES_DIR)
    random.shuffle(images_filenames)
    train = images_filenames[:TRAIN_SET_SIZE]
    validate = images_filenames[TRAIN_SET_SIZE:TRAIN_SET_SIZE+VALIDATION_SET_SIZE]
    assert len(validate) == VALIDATION_SET_SIZE

    filenames = TrainValidate()
    filenames.train.images = [os.path.join(IMAGES_DIR, name) for name in train]
    filenames.train.heatmaps = [os.path.join(HEATMAPS_DIR, name) for name in train]
    filenames.validate.images = filenames.train.images
    filenames.validate.heatmaps = filenames.train.heatmaps
    # filenames.validate.images = [os.path.join(IMAGES_DIR, name) for name in validate]
    # filenames.validate.heatmaps = [os.path.join(HEATMAPS_DIR, name) for name in validate]

    return filenames


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

        filenames = get_file_names()

        queue = filenames.map(lambda names: tf.train.string_input_producer(names, shuffle=False))

        image_reader = tf.WholeFileReader()

        file = queue.map(lambda q: image_reader.read(q)[1])

        bitmap = file \
            .map(lambda f: tf.image.decode_jpeg(f, ratio=2)) \
            .map(lambda i: tf.image.resize_images(i, [IMAGE_SIZE, IMAGE_SIZE]))

        for bm in [bitmap.train.images, bitmap.train.heatmaps, bitmap.validate.images, bitmap.validate.heatmaps]:
            bm.set_shape([IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

        def f_batch(bm):
            return tf.train.batch(
                [bm],
                batch_size=BATCH_SIZE,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * BATCH_SIZE
            )

        batch = bitmap.train.map(f_batch)

        pred = conv_net(batch.images)
        ground_truth = tf.div(batch.heatmaps, 256)

        tf.summary.image('image', tf.concat([batch.images, batch.heatmaps, pred], axis=2), max_outputs=BATCH_SIZE)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ground_truth))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        loss_summary = tf.summary.scalar('loss', cost)

        correct_pred = tf.equal(tf.argmax(pred, 3), tf.argmax(ground_truth, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        train_summaries = tf.summary.merge([loss_summary])

        def train_on_batch(sess):
            sess.run(optimizer)
            return sess.run([cost, accuracy, train_summaries])

        self.train_on_batch = train_on_batch

        summaries = tf.summary.merge_all()

        def run_all_summaries(sess, writer, gs):
            summ = sess.run(summaries)
            writer.add_summary(summ, gs)

        self.run_all_summaries = run_all_summaries

    def train(self):
        if not os.path.exists('save'):
            os.makedirs('save')

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                # for i in count():
                # self.run_all_summaries(sess, writer)

                for i in range(5):
                    for j in range(0, TRAIN_SET_SIZE // BATCH_SIZE):
                        loss, acc, summ = self.train_on_batch(sess)
                        if j % 20 == 0:
                            print("Iter " + str(j) + ", Minibatch Loss= " +
                                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                                  "{:.5f}".format(acc))
                        self.run_all_summaries(sess, writer, 20 * i + j)
                else:
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print("Optimization Finished!")
            finally:
                writer.close()
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    Model().train()
