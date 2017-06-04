"""
TODO should we use queues
TODO should we randomize batches
TODO do I leverage tensor shape type checking?
TODO name scope vs var scope
TODO For conv transpose use kernel size divisible by stride, or (possibly better option) upsample by nearest neighbour and then do regular convolution
TODO bias in conv nets - with/without batch norm

TODO conv heatmaps: give 3 dims, one hot = true
"""

from __future__ import print_function, generators

import datetime
import random

import tensorflow as tf
import os

LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
# DATA_SET_SIZE = 10593
DATA_SET_SIZE = 20
VALIDATION_SET_SIZE = 10
TRAIN_SET_SIZE = DATA_SET_SIZE - VALIDATION_SET_SIZE
DATA_DIR = os.environ['SPACENET'] or '/data/spacenet2/'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
HEATMAPS_DIR = os.path.join(DATA_DIR, 'heatmaps')
BATCH_SIZE = 3
IMAGE_SIZE = 256
CHANNELS = 3


def conv(features, in_channels, out_channels, kernel_size=3, name='conv'):
    with tf.name_scope(name):
        stride = 1
        weights = tf.get_variable("weights", [kernel_size, kernel_size, in_channels, out_channels],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        return tf.nn.conv2d(features, weights, strides=[1, stride, stride, 1], padding='SAME')


def conv_relu(features, in_channels, out_channels, name='conv_relu'):
    with tf.name_scope(name):
        return tf.nn.relu(conv(features, in_channels, out_channels))


def bn_conv_relu(features, in_channels, out_channels, name='bn_conv_relu'):
    with tf.name_scope(name):
        scale = tf.get_variable("scale", [in_channels],
                                initializer=tf.constant_initializer(1.0))
        offset = tf.get_variable("beta", [in_channels],
                                 initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(features, [0, 1, 2], keep_dims=True)
        out = tf.nn.batch_normalization(features, mean, variance, scale, offset, variance_epsilon=1e-4)
        return conv_relu(out, in_channels, out_channels)


def max_pool(features):
    return tf.nn.max_pool(features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # TODO maybe VALID


def bn_conv_relu_3_maxpool(features, channels, name='bn_conv_relu_3_maxpool'):
    with tf.name_scope(name):
        with tf.variable_scope('first'):
            out = bn_conv_relu(features, channels, channels)
        with tf.variable_scope('second'):
            out_skip_conn = bn_conv_relu(out, channels, channels)
        with tf.variable_scope('third'):
            out = bn_conv_relu(out_skip_conn, channels, channels)
        out = max_pool(out)
        return out_skip_conn, out


def upconv(features, in_channels, out_channels, out_size, name='upconv'):
    with tf.name_scope(name):
        stride = 2
        weights = tf.get_variable("weights", [3, 3, out_channels, in_channels],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        return tf.nn.conv2d_transpose(features, weights, output_shape=[BATCH_SIZE, out_size, out_size, 64], strides=[1, stride, stride, 1], padding='SAME')


def bn_upconv_relu(features, in_channels, out_channels, out_size, name='bn_upconv_relu'):
    with tf.name_scope(name):
        scale = tf.get_variable("scale", [in_channels],
                                initializer=tf.constant_initializer(1.0))
        offset = tf.get_variable("beta", [in_channels],
                                 initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(features, [0, 1, 2], keep_dims=True)
        out = tf.nn.batch_normalization(features, mean, variance, scale, offset, variance_epsilon=1e-4)
        return tf.nn.relu(upconv(out, in_channels, out_channels, out_size))


def concat(features_down, features_up, name='concat'):
    with tf.name_scope(name):
        return tf.concat([features_down, features_up], 3)


def concat_bn_conv_relu_2(features_down, features_up, name='concat_bn_conv_relu_2'):
    with tf.name_scope(name):
        out = concat(features_down, features_up)
        with tf.variable_scope('first'):
            out = bn_conv_relu(out, 128, 96)
        with tf.variable_scope('second'):
            return bn_conv_relu(out, 96, 64)


def concat_bn_conv_relu_2_bn_upconv_relu(features_down, features_up, out_size, name='concat_bn_conv_relu_2_bn_upconv_relu'):
    with tf.name_scope(name):
        out = concat_bn_conv_relu_2(features_down, features_up)
        return bn_upconv_relu(out, 64, 64, out_size)


def convout(features, in_channels, name='convout'):
    return conv(features, in_channels, 1, kernel_size=1, name=name)


def conv_net(features):
    with tf.variable_scope('down1'):
        features_down1 = conv_relu(features, 3, 64)

    with tf.variable_scope('down2'):
        with tf.variable_scope('first'):
            features_down2_skip_conn = bn_conv_relu(features_down1, 64, 64)
        with tf.variable_scope('second'):
            features_down2 = bn_conv_relu(features_down2_skip_conn, 64, 64)
        features_down2 = max_pool(features_down2)

    with tf.variable_scope('down3'):
        features_down3_skip_conn, features_down3 = bn_conv_relu_3_maxpool(features_down2, 64)
    with tf.variable_scope('down4'):
        features_down4_skip_conn, features_down4 = bn_conv_relu_3_maxpool(features_down3, 64)
    with tf.variable_scope('down5'):
        features_down5_skip_conn, features_down5 = bn_conv_relu_3_maxpool(features_down4, 64)
    with tf.variable_scope('down6'):
        features_down6_skip_conn, features_down6 = bn_conv_relu_3_maxpool(features_down5, 64)

    with tf.variable_scope('up1'):
        with tf.variable_scope('first'):
            features_up1 = bn_conv_relu(features_down6, 64, 64)
        with tf.variable_scope('second'):
            features_up1 = bn_conv_relu(features_up1, 64, 64)
        features_up1 = bn_upconv_relu(features_up1, 64, 64, 16)

    with tf.variable_scope('up2'):
        features_up2 = concat_bn_conv_relu_2_bn_upconv_relu(features_down6_skip_conn, features_up1, 32)
    with tf.variable_scope('up3'):
        features_up3 = concat_bn_conv_relu_2_bn_upconv_relu(features_down5_skip_conn, features_up2, 64)
    with tf.variable_scope('up4'):
        features_up4 = concat_bn_conv_relu_2_bn_upconv_relu(features_down4_skip_conn, features_up3, 128)
    with tf.variable_scope('up5'):
        features_up5 = concat_bn_conv_relu_2_bn_upconv_relu(features_down3_skip_conn, features_up4, 256)

    with tf.variable_scope('up6'):
        features_up6 = concat_bn_conv_relu_2(features_down2_skip_conn, features_up5)
        features_up6 = convout(features_up6, 64)

    return features_up6


def read_and_decode(folder, filename):
    path = os.path.join(folder, filename)
    with open(path) as f:
        return tf.image.decode_jpeg(f.read(), ratio=2)


def create_partition_vector():
    partitions = [0] * DATA_SET_SIZE
    partitions[:VALIDATION_SET_SIZE] = [1] * VALIDATION_SET_SIZE
    random.shuffle(partitions)
    return partitions


def setup():
    images_before_resizing = []
    heatmaps_before_resizing = []

    images_filenames = sorted(os.listdir(IMAGES_DIR))
    heatmaps_filenames = sorted(os.listdir(HEATMAPS_DIR))
    data = zip(range(DATA_SET_SIZE), images_filenames, heatmaps_filenames)

    for _, image_name, heatmap_name in data:
        images_before_resizing.append(read_and_decode(IMAGES_DIR, image_name))
        heatmaps_before_resizing.append(read_and_decode(HEATMAPS_DIR, heatmap_name))
    images_before_resizing = tf.stack(images_before_resizing)
    heatmaps_before_resizing = tf.stack(heatmaps_before_resizing)

    images = tf.image.resize_images(images_before_resizing, [IMAGE_SIZE, IMAGE_SIZE])
    heatmaps = tf.image.resize_images(heatmaps_before_resizing, [IMAGE_SIZE, IMAGE_SIZE])

    with tf.name_scope('input'):
        partitions = create_partition_vector()

        train_images_value, validate_images_value = tf.dynamic_partition(images, partitions, 2)
        train_heatmaps_value, validate_heatmaps_value = tf.dynamic_partition(heatmaps, partitions, 2)

        def data_var(init):
            return tf.Variable(init, trainable=False, validate_shape=False)

        train_images = data_var(train_images_value)
        train_heatmaps = data_var(train_heatmaps_value)
        validate_images = data_var(validate_images_value)
        validate_heatmaps = data_var(validate_heatmaps_value)

        train_images.set_shape([TRAIN_SET_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        train_heatmaps.set_shape([TRAIN_SET_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        validate_images.set_shape([VALIDATION_SET_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        validate_heatmaps.set_shape([VALIDATION_SET_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

        batch_start = tf.placeholder(tf.int32, shape=[])
        batch_images = tf.slice(train_images, [batch_start, 0, 0, 0], [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        batch_heatmaps = tf.slice(train_heatmaps, [batch_start, 0, 0, 0], [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

        graph = conv_net(batch_images)

        with tf.Session() as sess:
            # writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            tf.initialize_all_variables().run()

            sess.run(graph, feed_dict={batch_start: 1})

            # summary = sess.run(image_summaries('second'), feed_dict={batch_start: 8})
            # writer.add_summary(summary)
            # writer.close()


if __name__ == '__main__':
    setup()



# # Construct model
# pred = conv_net(x)
#
# # Define loss and optimizer
# # Applies softmax internally
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


