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
BATCH_SIZE = int(os.environ.get('BATCH_SIZE') or 2)
AUGMENTED_BATCH_SIZE = 8 * BATCH_SIZE
HALF_IMAGE_SIZE = 325
IMAGE_SIZE = 256
CHANNELS = 3
LEARNING_RATE = 0.001
IMAGE_TRANSFORMATION_NUMBER = 8


def conv(features, in_channels, out_channels, kernel_size=3, name='conv'):
    with tf.variable_scope(name):
        stride = 1
        weights = tf.get_variable("weights", [kernel_size, kernel_size, in_channels, out_channels],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", shape=[out_channels], initializer=tf.random_normal_initializer(stddev=0.1))
        out = tf.nn.conv2d(features, weights, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(out, bias)


def conv_relu(features, in_channels, out_channels, name='conv_relu'):
    with tf.name_scope(name):
        return tf.nn.relu(conv(features, in_channels, out_channels))


def bn_conv_relu(features, in_channels, out_channels, name='bn_conv_relu'):
    with tf.variable_scope(name):
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
    with tf.variable_scope(name):
        out = bn_conv_relu(features, channels, channels, name='first')
        out_skip_conn = bn_conv_relu(out, channels, channels, name='second')
        out = bn_conv_relu(out_skip_conn, channels, channels, name='third')
        out = max_pool(out)
        return out_skip_conn, out


def upconv(features, in_channels, out_channels, out_size, name='upconv'):
    with tf.variable_scope(name):
        stride = 2
        weights = tf.get_variable("weights", [4, 4, out_channels, in_channels],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        return tf.nn.conv2d_transpose(
            features, weights,
            output_shape=[tf.shape(features)[0], out_size, out_size, 64],
            strides=[1, stride, stride, 1],
            padding='SAME')


def bn_upconv_relu(features, in_channels, out_channels, out_size, name='bn_upconv_relu'):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale", [in_channels],
                                initializer=tf.constant_initializer(1.0))
        offset = tf.get_variable("beta", [in_channels],
                                 initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(features, [0, 1, 2], keep_dims=True)
        out = tf.nn.batch_normalization(features, mean, variance, scale, offset, variance_epsilon=1e-4)
        return tf.nn.relu(upconv(out, in_channels, out_channels, out_size))


def concat(features_down, features_up, name='concat'):
    with tf.variable_scope(name):
        return tf.concat([features_down, features_up], axis=3)


def concat_bn_conv_relu_2(features_down, features_up, name='concat_bn_conv_relu_2'):
    with tf.variable_scope(name):
        out = concat(features_down, features_up)
        out = bn_conv_relu(out, 128, 96, name='first')
        return bn_conv_relu(out, 96, 64, name='second')


def concat_bn_conv_relu_2_bn_upconv_relu(features_down, features_up, out_size,
                                         name='concat_bn_conv_relu_2_bn_upconv_relu'):
    with tf.variable_scope(name):
        out = concat_bn_conv_relu_2(features_down, features_up)
        return bn_upconv_relu(out, 64, 64, out_size)


def convout(features, in_channels, name='convout'):
    return conv(features, in_channels, 3, kernel_size=1, name=name)


def conv_net(features):
    with tf.variable_scope('down1'):
        features_down1 = conv_relu(features, 3, 64)

    with tf.variable_scope('down2'):
        features_down2_skip_conn = bn_conv_relu(features_down1, 64, 64, name='first')
        features_down2 = bn_conv_relu(features_down2_skip_conn, 64, 64, name='second')
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
        features_up1 = bn_conv_relu(features_down6, 64, 64, name='first')
        features_up1 = bn_conv_relu(features_up1, 64, 64, name='second')
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

    train_images_names = [os.path.join(IMAGES_DIR, name) for name in train]
    train_heatmaps_names = [os.path.join(HEATMAPS_DIR, name) for name in train]
    validate_images_names = [os.path.join(IMAGES_DIR, name) for name in validate]
    validate_heatmaps_names = [os.path.join(HEATMAPS_DIR, name) for name in validate]

    return train_images_names, train_heatmaps_names, validate_images_names, validate_heatmaps_names


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

        train_images_names, train_heatmaps_names, validate_images_names, validate_heatmaps_names = get_file_names()

        train_images_queue = tf.train.string_input_producer(train_images_names, shuffle=False)
        train_heatmaps_queue = tf.train.string_input_producer(train_heatmaps_names, shuffle=False)
        validate_images_queue = tf.train.string_input_producer(validate_images_names, shuffle=False)
        validate_heatmaps_queue = tf.train.string_input_producer(validate_heatmaps_names, shuffle=False)

        image_reader = tf.WholeFileReader()

        _, train_image_file = image_reader.read(train_images_queue)
        _, train_heatmap_file = image_reader.read(train_heatmaps_queue)
        _, validate_image_file = image_reader.read(validate_images_queue)
        _, validate_heatmap_file = image_reader.read(validate_heatmaps_queue)

        train_image = tf.image.decode_jpeg(train_image_file, ratio=2)
        train_heatmap = tf.image.decode_jpeg(train_heatmap_file, ratio=2)
        validate_image = tf.image.decode_jpeg(train_image_file, ratio=2)
        validate_heatmap = tf.image.decode_jpeg(train_image_file, ratio=2)

        train_image = tf.image.resize_images(train_image, [IMAGE_SIZE, IMAGE_SIZE])
        train_heatmap = tf.image.resize_images(train_heatmap, [IMAGE_SIZE, IMAGE_SIZE])
        validate_image = tf.image.resize_images(validate_image, [IMAGE_SIZE, IMAGE_SIZE])
        validate_heatmap = tf.image.resize_images(validate_heatmap, [IMAGE_SIZE, IMAGE_SIZE])

        train_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        train_heatmap.set_shape([IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        validate_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        validate_heatmap.set_shape([IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

        # num_preprocess_threads = 2
        # min_queue_examples = 64
        # batch_images = tf.train.batch(
        #     [train_image],
        #     batch_size=BATCH_SIZE,
        #     num_threads=num_preprocess_threads,
        #     capacity=min_queue_examples + 3 * BATCH_SIZE)
        #
        # batch_heatmaps = tf.train.batch(
        #     [train_heatmap],
        #     batch_size=BATCH_SIZE,
        #     num_threads=num_preprocess_threads,
        #     capacity=min_queue_examples + 3 * BATCH_SIZE)

        # batch of size 1
        batch_images = tf.expand_dims(train_image, 0)
        batch_heatmaps = tf.expand_dims(train_heatmap, 0)

        augmented_batch_images = augment_many(batch_images)
        augmented_batch_heatmaps = augment_many(batch_heatmaps)

        pred = conv_net(augmented_batch_images)
        ground_truth = tf.div(augmented_batch_heatmaps, 256)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ground_truth))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 3), tf.argmax(ground_truth, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        def train_on_batch(sess):
            sess.run(optimizer)
            return sess.run([cost, accuracy])

        self.train_on_batch = train_on_batch

        # validation

        tf.get_variable_scope().reuse_variables()

        # batch_validate_images = tf.train.batch(
        #     [validate_image],
        #     batch_size=1,
        #     num_threads=num_preprocess_threads,
        #     capacity=min_queue_examples + 3)
        #
        # batch_validate_heatmaps = tf.train.batch(
        #     [validate_heatmap],
        #     batch_size=1,
        #     num_threads=num_preprocess_threads,
        #     capacity=min_queue_examples + 3)

        # batch of size 1
        batch_validate_images = tf.expand_dims(validate_image, 0)
        batch_validate_heatmaps = tf.expand_dims(validate_heatmap, 0)

        validation_pred = gather_transformations(conv_net(augment_many(batch_validate_images)))
        validation_ground_truth = tf.div(batch_validate_heatmaps, 256)

        validation_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=validation_pred,
            labels=validation_ground_truth))
        validation_correct_pred = tf.equal(tf.argmax(validation_pred, 3), tf.argmax(validation_ground_truth, 3))
        validation_accuracy = tf.reduce_mean(tf.cast(validation_correct_pred, tf.float32))

        validation_correct_pred_all = tf.reduce_mean([validation_cost for _ in range(VALIDATION_SET_SIZE)])
        validation_accuracy_all = tf.reduce_mean([validation_accuracy for _ in range(VALIDATION_SET_SIZE)])

        def validate(sess):
            loss, acc = sess.run([validation_correct_pred_all, validation_accuracy_all])
            print("Validation loss %g" % loss)
            print("Validation accuracy %g" % acc)

        self.validate = validate

    def train(self):
        saver = tf.train.Saver(tf.trainable_variables())
        if not os.path.exists('save'):
            os.makedirs('save')

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for i in count():
                    for j in range(0, TRAIN_SET_SIZE // BATCH_SIZE):
                        loss, acc = self.train_on_batch(sess)
                        if j % 20:
                            print("Iter " + str(j) + ", Minibatch Loss= " +
                                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                                  "{:.5f}".format(acc))

                    # validate after every five epochs
                    if i % 5 == 0:
                        self.validate(sess)

                    if i % 10 == 0:
                        saver.save(sess, 'save/model', global_step=i)

                    # writer.add_summary(summaries)
                    # writer.close()
            except KeyboardInterrupt:
                # TODO save model
                print("Optimization Finished!")
                self.validate(sess)
                saver.save(sess, 'save/model', global_step=0)
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    Model().train()
