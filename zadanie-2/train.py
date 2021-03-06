from __future__ import print_function, generators

import datetime
import random
from itertools import count

import tensorflow as tf
import os

LOG_DIR = 'logs/' + datetime.datetime.now().strftime("%B-%d-%Y;%H:%M")
DATA_SET_SIZE = int(os.environ.get('DATA_SET_SIZE') or 10593)
VALIDATION_SET_SIZE = int(os.environ.get('VALIDATION_SET_SIZE') or 593)
TRAIN_SET_SIZE = DATA_SET_SIZE - VALIDATION_SET_SIZE
DATA_DIR = os.environ.get('SPACENET') or '/data/spacenet2/'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
HEATMAPS_DIR = os.path.join(DATA_DIR, 'heatmaps')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE') or 10)
AUGMENTED_BATCH_SIZE = 8 * BATCH_SIZE
IMAGE_SIZE = 256
CHANNELS = 3
LEARNING_RATE = 0.001
IMAGE_TRANSFORMATION_NUMBER = 8


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
        return tf.nn.conv2d_transpose(
            features, weights,
            output_shape=[tf.shape(features)[0], out_size, out_size, 64],
            strides=[1, stride, stride, 1],
            padding='SAME')


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


def concat_bn_conv_relu_2_bn_upconv_relu(features_down, features_up, out_size,
                                         name='concat_bn_conv_relu_2_bn_upconv_relu'):
    with tf.name_scope(name):
        out = concat_bn_conv_relu_2(features_down, features_up)
        return bn_upconv_relu(out, 64, 64, out_size)


def convout(features, in_channels, name='convout'):
    return conv(features, in_channels, 3, kernel_size=1, name=name)


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


def read_image(folder, filename):
    path = os.path.join(folder, filename)
    with open(path) as f:
        return f.read()


def create_partition_vector():
    partitions = [0] * DATA_SET_SIZE
    partitions[:VALIDATION_SET_SIZE] = [1] * VALIDATION_SET_SIZE
    random.shuffle(partitions)
    return partitions


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
    return tf.reduce_mean(revert_transformations(augmented_images), 0)


def augment_many(images):
    augmented_batch = tf.map_fn(augment_image, images)
    return tf.reshape(augmented_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])


def read_images():
    images = []
    heatmaps = []

    images_filenames = sorted(os.listdir(IMAGES_DIR))
    # heatmaps_filenames = sorted(os.listdir(HEATMAPS_DIR))
    data = zip(range(DATA_SET_SIZE), images_filenames)

    for _, filename in data:
        images.append(read_image(IMAGES_DIR, filename))
        heatmaps.append(read_image(HEATMAPS_DIR, filename))

    return images, heatmaps


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

        with tf.name_scope('input'):
            images_initializer = tf.placeholder(dtype=tf.string, shape=[DATA_SET_SIZE])
            heatmaps_initializer = tf.placeholder(dtype=tf.string, shape=[DATA_SET_SIZE])

            def decode(image):
                return tf.image.decode_jpeg(image, ratio=2)

            images_before_resizing = tf.map_fn(decode, images_initializer, dtype=tf.uint8)
            heatmaps_before_resizing = tf.map_fn(decode, heatmaps_initializer, dtype=tf.uint8)

            images = tf.image.resize_images(images_before_resizing, [IMAGE_SIZE, IMAGE_SIZE])
            heatmaps = tf.image.resize_images(heatmaps_before_resizing, [IMAGE_SIZE, IMAGE_SIZE])

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

            validate_images_augmented = augment_many(validate_images)

        def initialize_images(sess, images, heatmaps):
            # images_vars = [train_images, train_heatmaps, validate_images, validate_heatmaps]
            # sess.run(
            #     [var.initializer for var in images_vars],
            #     feed_dict={images_initializer: images, heatmaps_initializer: heatmaps})
            sess.run(tf.global_variables_initializer(),
                     feed_dict={images_initializer: images, heatmaps_initializer: heatmaps})

        self.initialize_images = initialize_images

        with tf.name_scope('batch'):
            batch_start = tf.placeholder(tf.int32, shape=[])
            batch_images = tf.slice(train_images, [batch_start, 0, 0, 0],
                                    [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
            batch_heatmaps = tf.slice(train_heatmaps, [batch_start, 0, 0, 0],
                                      [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

            augmented_batch_images = augment_many(batch_images)
            augmented_batch_heatmaps = augment_many(batch_heatmaps)

        pred = conv_net(augmented_batch_images)
        ground_truth = tf.div(augmented_batch_heatmaps, 256)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ground_truth))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 3), tf.argmax(ground_truth, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        def train_on_batch(sess, batch_begin):
            sess.run(optimizer, feed_dict={batch_start: batch_begin})
            return sess.run([cost, accuracy], feed_dict={batch_start: batch_begin})

        self.train_on_batch = train_on_batch

        # validation

        tf.get_variable_scope().reuse_variables()

        validation_pred = conv_net(validate_images_augmented)
        validation_pred = tf.reshape(
            validation_pred,
            [-1, IMAGE_TRANSFORMATION_NUMBER, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        validation_pred = tf.map_fn(gather_transformations, validation_pred)
        validation_ground_truth = tf.div(validate_heatmaps, 256)
        validation_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=validation_pred,
            labels=validation_ground_truth))
        validation_correct_pred = tf.equal(tf.argmax(validation_pred, 3), tf.argmax(validation_ground_truth, 3))
        validation_accuracy = tf.reduce_mean(tf.cast(validation_correct_pred, tf.float32))

        def validate(sess):
            loss, acc = sess.run([validation_cost, validation_accuracy])
            print("Validation loss %g" % loss)
            print("Validation accuracy %g" % acc)

        self.validate = validate

    def train(self):
        images, heatmaps = read_images()

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            self.initialize_images(sess, images, heatmaps)
            # sess.run(tf.variables_initializer(tf.trainable_variables()))

            try:
                for i in count():
                    for j, batch_begin in enumerate(range(0, TRAIN_SET_SIZE - BATCH_SIZE, BATCH_SIZE)):
                        loss, acc = self.train_on_batch(sess, batch_begin)
                        print("Iter " + str(j) + ", Minibatch Loss= " +
                              "{:.6f}".format(loss) + ", Training Accuracy= " +
                              "{:.5f}".format(acc))

                    # validate after every epoch
                    self.validate(sess)
                    saver.save(sess, 'save/model', global_step=i)

                    # writer.add_summary(summaries)
                    # writer.close()
            except KeyboardInterrupt:
                # TODO save model
                print("Optimization Finished!")
                self.validate(sess)
                saver.save(sess, 'save/model', global_step=0)


if __name__ == '__main__':
    Model().train()
