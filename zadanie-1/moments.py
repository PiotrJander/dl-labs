from datetime import datetime
import tensorflow as tf
import numpy as np


def moments(x, axes, keep_dims=True):
    """
    TODO make axes a list rather than number
    """
    with tf.name_scope("my_moments"):
        mean = tf.reduce_mean(x, axes, keep_dims=keep_dims)
        x_minus_mean = x - mean
        x_minus_mean_squared = tf.square(x_minus_mean)
        variance = tf.reduce_mean(x_minus_mean_squared, axes, keep_dims=keep_dims)
        return mean, variance


def test_moments():
    LOG_DIR = 'out/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    a_value = np.array([[2, 3], [4, 5]])

    a = tf.placeholder(dtype=tf.float32)
    tfmoms = tf.nn.moments(a, [1], keep_dims=True)
    mymoms = moments(a, 1, keep_dims=True)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        summary_writer.flush()

        (mymean, myvar), (tfmean, tfvar) = sess.run([mymoms, tfmoms], feed_dict={a: a_value})
        print("mean: ", mymean, tfmean)
        print("variance: ", myvar, tfvar)
