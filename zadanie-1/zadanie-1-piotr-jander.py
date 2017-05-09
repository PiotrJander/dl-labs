'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
# Small epsilon value for the BN transform
epsilon = 1e-3

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


def conv_relu_maxpool(input, kernel_shape, bias_shape, strides=1, k=2):
    """
    Creates three layers: conv, relu, maxpool
    """
    # Create variables
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.random_normal_initializer())

    # Convolution Layer
    input = tf.nn.conv2d(input, weights, strides=[1, strides, strides, 1], padding='SAME')
    input = tf.nn.relu(input + biases)

    # Max Pooling (down-sampling)
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def fully_conn(input, matrix_shape, bias_shape):
    """
    Creates a fully connected layer
    """
    # Creates variables
    weights = tf.get_variable("weights", matrix_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.random_normal_initializer())

    # Fully connected layer
    return tf.add(tf.matmul(input, weights), biases)


# Create model
def conv_net(input):
    # Reshape input picture
    input = tf.reshape(input, shape=[-1, 28, 28, 1])

    # Convolutional layers
    with tf.variable_scope("conv_1"):
        conv1 = conv_relu_maxpool(input, [5, 5, 1, 32], [32])
    with tf.variable_scope("conv_2"):
        conv2 = conv_relu_maxpool(conv1, [5, 5, 32, 64], [64])

    # Fully connected layer
    with tf.variable_scope("hidden_1"):
        fc1_inputs_num = 7 * 7 * 64
        fc1 = tf.reshape(conv2, [-1, fc1_inputs_num])
        fc1 = fully_conn(fc1, [fc1_inputs_num, 1024], [1024])
        fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    with tf.variable_scope("out"):
        out = fully_conn(fc1, [1024, n_classes], [n_classes])

    return out


# Construct model
pred = conv_net(x)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                        y: mnist.test.labels[:256]}))
