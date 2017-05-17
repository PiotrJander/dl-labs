import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

''''
Task 1. Write simple RNN to recognize MNIST digits.
The image is 28x28. Flatten it to a 784 vector.
Pick some divisior d of 784, e.g. d = 28. 
At each timestep the input will be d bits of the image. 
Thus the sequence length will be 784 / d
You should be able to get over 93% accuracy
Write your own implementation of RNN, you can look at the one from the slide,
but do not copy it blindly.

Task 2. 
Same, but use LSTM instead of simple RNN.
What accuracy do you get.
Experiment with choosing d, compare RNN and LSTM.
Again do not use builtin Tensorflow implementation. Write your own :)

Task 3*.
Make LSTM a deep bidirectional, multilayer LSTM.
'''


class MnistTrainer(object):
    def train_on_batch(self, batch_xs, batch_ys):
        raise NotImplementedError()
        ### WRITE YOUR CODE HERE ###

    def create_model(self):
        """
        ok let's start here
        we will train on sequence of 28 rows of pixels
        so we take a sequence of fixed length
        init state can be randomly initialized, or zero vector
        input vector is 28 dim
        decide how big state vector should be (784 maybe?)
        recurrent layer is a function of state vec and input vec
        and it produces output vector and state vec
        in mnist, we only produce output after consuming all input vectors
        so maybe output is the network's current guess about the classification

        ok what about making it simple and learning to predict characters over alphabet "helo"
        trained on the sample "hello"?
        """
        # d = 28
        # steps_n = 784 / d
        #
        # self.x = tf.placeholder(dtype=tf.float32, shape=[None, steps_n, d])
        # self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        # ### WRITE YOUR CODE HERE ###

        # TODO init self.h state to zero
        pass

        # input: 1-of-5 vector (also EOF)
        # output: same
        # state: m-dim vector

        # easy: identity
        # predict same char next
        # encode input as matrix, feed rows/columns one by one
        # gather net output and compute error function
        # in real world softmax
        # we use unstack on get list on tensors (rows)
        # we need some sort of mapFold

        # what is state? not variable, because we don't optimize it in gradient descent
        # not constant, because we change it
        # perhaps placeholder, because we pass it as arg

        # that's how we train on batch in zadanie1

        # # Construct model
        # pred = conv_net(x)
        #
        # # Define loss and optimizer
        # # Applies softmax internally
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        #
        # sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # the conv_net in zadanie1 takes a tensor (list of images)
        # and returns a tensor (list of outputs from last linear layer, prob shape n x 10)

        # simple: take just one example at a time


    def cell(self, x):
        # get variables
        shape = [5, 5]
        w_hh = tf.get_variable("w_hh", shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_xh = tf.get_variable("w_xh", shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_hy = tf.get_variable("w_hy", shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.1))

        # TODO bias
        # update the hidden state
        self.h = tf.tanh(tf.matmult(self.W_hh, self.h) + tf.matmult(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y

    def __init__(self):
        super(MnistTrainer, self).__init__()
        self.h = [0, 0, 0, 0, 0]  # TODO tf tensor

    def train(self):

        self.create_model()
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        with tf.Session() as self.sess:
            tf.global_variables_initializer().run()  # initialize variables
            batches_n = 100000
            mb_size = 128

            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)

                    loss, accuracy = self.train_on_batch(batch_xs, batch_ys)

                    losses.append(loss)

                    if batch_idx % 100 == 0:
                        print('Batch {batch_idx}: loss {loss}, mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, loss=loss, mean_loss=np.mean(losses[-200:]))
                        )


            except KeyboardInterrupt:
                print('Stopping training!')
                pass

                # Test trained model
                ### WRITE YOUR CODE HERE ###


if __name__ == '__main__':
    trainer = MnistTrainer()
    trainer.train()
