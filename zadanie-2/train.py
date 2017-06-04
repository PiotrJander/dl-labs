"""
TODO should we use queues
TODO should we randomize batches
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
IMAGES_DIR = '/data/spacenet2/images/'
HEATMAPS_DIR = '/data/spacenet2/heatmaps/'


def train_set_size():
    return DATA_SET_SIZE - VALIDATION_SET_SIZE


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
    images = []
    heatmaps = []

    images_filenames = sorted(os.listdir(IMAGES_DIR))
    heatmaps_filenames = sorted(os.listdir(HEATMAPS_DIR))
    data = zip(range(DATA_SET_SIZE), images_filenames, heatmaps_filenames)

    for _, image_name, heatmap_name in data:
        images.append(read_and_decode(IMAGES_DIR, image_name))
        heatmaps.append(read_and_decode(HEATMAPS_DIR, heatmap_name))
    images = tf.stack(images)
    heatmaps = tf.stack(heatmaps)

    with tf.name_scope('input'):
        # train_shape = [train_set_size(), 325, 325, 3]
        # validate_shape = [VALIDATION_SET_SIZE, 325, 325, 3]
        partitions = create_partition_vector()

        train_images_value, validate_images_value = tf.dynamic_partition(images, partitions, 2)
        train_heatmaps_value, validate_heatmaps_value = tf.dynamic_partition(heatmaps, partitions, 2)

        # train_images_value1 = tf.constant(
        #     train_images_value,
        #     dtype=tf.uint8, shape=)

        def data_var(data):
            return tf.Variable(data, trainable=False, validate_shape=False)

        train_images = data_var(train_images_value)
        train_heatmaps = data_var(train_heatmaps_value)
        validate_images = data_var(validate_images_value)
        validate_heatmaps = data_var(validate_heatmaps_value)

        # images_summary_op = tf.summary.image("train_images", train_images[:10])
        # heatmaps_summary_op = tf.summary.image("train_heatmaps", train_heatmaps[:10])
        # all_summaries = tf.summary.merge_all()
        image_summaries = tf.summary.merge(tf.summary.image(str(i), tensor[:3])
                                           for i, tensor
                                           in enumerate([train_images, train_heatmaps, validate_images, validate_heatmaps])
                                           )

        # Where to take the slice for the batch
        # batch_start = tf.placeholder(tf.uint16)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            tf.initialize_all_variables().run()

            summary = sess.run(image_summaries)
            writer.add_summary(summary)
            writer.close()


if __name__ == '__main__':
    setup()









