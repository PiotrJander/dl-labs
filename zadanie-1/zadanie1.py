"""
TODO opis

Attribution:
1.
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

2. Implementing Batch Normalization in Tensorflow
http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
"""

from __future__ import print_function

from datetime import datetime
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
learning_rate = 0.001
training_iters = 20000
# training_iters = 5000
batch_size = 50
display_step = 100
epoch_size = 1000
# Small epsilon value for the BN transform
epsilon = 1e-4

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# ~~~ moments

# def my_moments(x, axes):
#     """
#     TODO make axes a list rather than number
#     """
#     with tf.name_scope("my_moments"):
#         mean = tf.reduce_mean(x, axes[0], keep_dims=True)
#         x_minus_mean = x - mean
#         x_minus_mean_squared = tf.square(x_minus_mean)
#         variance = tf.reduce_mean(x_minus_mean_squared, axes[0], keep_dims=True)
#         return mean, variance
#
# def my_moments(x, axes):
#     """
#     Replaces tf.nn.moments
#     """
#     with tf.name_scope("my_moments"):
#         mean = my_reduce_mean(x, axes)
#         x_minus_mean = x - mean
#         x_minus_mean_squared = tf.square(x_minus_mean)
#         variance = my_reduce_mean(x_minus_mean_squared, axes)
#         return mean, variance
#
#
# def my_reduce_mean(x, axes):
#     """
#     Takes a list of axes.
#     """
#     mean = x
#     for axis in axes:
#         mean = tf.reduce_mean(mean, axis, keep_dims=True)
#     return mean


# def conv_relu_maxpool(input, kernel_shape, bias_shape, strides=1, k=2):
#     """
#     Creates three layers: conv, relu, maxpool
#     """
#     # Create variables
#     weights = tf.get_variable("weights", kernel_shape,
#                               initializer=tf.truncated_normal_initializer(stddev=0.1))
#     biases = tf.get_variable("biases", bias_shape,
#                              initializer=tf.constant_initializer(0.1))
#
#     # Convolution Layer
#     z = tf.nn.conv2d(input, weights, strides=[1, strides, strides, 1],
#                      padding='SAME')
#     out = tf.nn.relu(z + biases)
#
#     # Max Pooling (down-sampling)
#     return tf.nn.max_pool(out, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                           padding='SAME')


def conv_relu_maxpool_batch_norm(input, kernel_shape, bias_shape, strides=1, k=2):
    """
    Creates three layers: conv, relu, maxpool
    """
    # Create variables
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # Create scale and beta (shift) params
    scale1 = tf.get_variable("scale", bias_shape,
                             initializer=tf.constant_initializer(1.0))
    beta1 = tf.get_variable("beta", bias_shape,
                            initializer=tf.constant_initializer(0.0))

    # Convolution Layer
    z = tf.nn.conv2d(input, weights, strides=[1, strides, strides, 1],
                     padding='SAME')

    # Calculate batch mean and variance
    # batch_mean1, batch_var1 = my_moments(z, [0, 1, 2])
    batch_mean1, batch_var1 = tf.nn.moments(z, [0, 1, 2], keep_dims=True)

    # Apply the initial batch normalizing transform
    z1_hat = (z - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

    # Scale and shift to obtain the final output of the batch normalization
    # this value is fed into the activation function (here a sigmoid)
    out = scale1 * z1_hat + beta1

    # Max Pooling (down-sampling)
    return tf.nn.max_pool(out, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def fully_conn(input, matrix_shape, bias_shape):
    """
    Creates a fully connected layer
    """
    # Creates variables
    weights = tf.get_variable("weights", matrix_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.1))

    # Fully connected layer
    return tf.matmul(input, weights) + biases


def fully_conn_batch_norm(input, matrix_shape, bias_shape):
    """
    Creates a fully connected layer
    """
    # Creates weight variables
    weights = tf.get_variable("weights", matrix_shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # Create scale and beta (shift) params
    scale1 = tf.get_variable("scale", bias_shape,
                             initializer=tf.constant_initializer(1.0))
    beta1 = tf.get_variable("beta", bias_shape,
                            initializer=tf.constant_initializer(0.0))

    # Fully connected layer
    z = tf.matmul(input, weights)

    # Calculate batch mean and variance
    # batch_mean1, batch_var1 = my_moments(z, [0])
    batch_mean1, batch_var1 = tf.nn.moments(z, [0], keep_dims=True)

    # Apply the initial batch normalizing transform
    z1_hat = (z - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

    # Scale and shift to obtain the final output of the batch normalization
    # this value is fed into the activation function (here a sigmoid)
    return scale1 * z1_hat + beta1


# Create model
def conv_net(input):
    # Reshape input picture
    input = tf.reshape(input, shape=[-1, 28, 28, 1])

    # Convolutional layers
    with tf.variable_scope("conv_1"):
        conv1 = conv_relu_maxpool_batch_norm(input, [5, 5, 1, 32], [32])
        # conv1 = conv_relu_maxpool(input, [5, 5, 1, 32], [32])

    with tf.variable_scope("conv_2"):
        conv2 = conv_relu_maxpool_batch_norm(conv1, [5, 5, 32, 64], [64])
        # conv2 = conv_relu_maxpool(conv1, [5, 5, 32, 64], [64])

    # Fully connected layer
    with tf.variable_scope("hidden_1"):
        fc1_inputs_num = 7 * 7 * 64
        fc1 = tf.reshape(conv2, [-1, fc1_inputs_num])
        fc1 = fully_conn_batch_norm(fc1, [fc1_inputs_num, 1024], [1024])
        # fc1 = fully_conn(fc1, [fc1_inputs_num, 1024], [1024])
        fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    with tf.variable_scope("out"):
        out = fully_conn(fc1, [1024, n_classes], [n_classes])

    return out


# Construct model
pred = conv_net(x)

# Define loss and optimizer
# Applies softmax internally
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

if __name__ == '__main__':
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        validation_results = []

        for i in range(training_iters):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if i % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y})
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            # vaildate at the end of every epoch
            if i % epoch_size == 0:
                validation_acc = accuracy.eval(feed_dict={x: mnist.validation.images,
                                                          y: mnist.validation.labels})
                validation_results.append(validation_acc)
                print("Validation accuracy %g" % validation_acc)

                if validation_acc > 99.1:
                    break

        print("Optimization Finished!")
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y: mnist.test.labels}))
        print(validation_results)
