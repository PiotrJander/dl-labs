import tensorflow as tf


def my_moments(x, axes):
    """
    TODO make axes a list rather than number
    """
    with tf.name_scope("my_moments"):
        mean = my_reduce_mean(x, axes)
        x_minus_mean = x - mean
        x_minus_mean_squared = tf.square(x_minus_mean)
        variance = my_reduce_mean(x_minus_mean_squared, axes)
        return mean, variance


def my_reduce_mean(x, axes):
    """
    Takes a list of axes.
    """
    mean = x
    for axis in axes:
        mean = tf.reduce_mean(mean, axis, keep_dims=True)
    return mean


def test_my_reduce_mean():
    # og_dir = 'out/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    a_value = [[[2, 4], [4, 6]], [[5, 7], [1, 3]]]

    a = tf.placeholder(dtype=tf.float32)
    mymean = my_reduce_mean(a, [1, 2])
    with tf.Session() as sess:
        # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # summary_writer.flush()

        mymean = sess.run(mymean, feed_dict={a: a_value})
        print(mymean)


# def test_my_moments():
#     log_dir = 'out/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     a_value = [[2, 3], [4, 5]]
#
#     a = tf.placeholder(dtype=tf.float32)
#     tfmoms = tf.nn.moments(a, [1])
#     mymoms = my_moments(a, [1], keep_dims=True)
#     with tf.Session() as sess:
#         summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
#         summary_writer.flush()
#
#         (mymean, myvar), (tfmean, tfvar) = sess.run([mymoms, tfmoms], feed_dict={a: a_value})
#         print("mean: ", mymean, tfmean)
#         print("variance: ", myvar, tfvar)
