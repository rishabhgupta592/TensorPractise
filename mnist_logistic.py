
# Linear regression
# MNIST
# Each image is 28 pixels by 28 pixels = 28*28 = 784 input neurons
# 10 classes output

# Input array will be no of data rows (n) and 784 columns.
# y = x.w + b
# [10, 1] = [n, 784] * [784, 10] +  [10, 1]

import tensorflow as tf

# Data downloader
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Input matrix
x = tf.placeholder(tf.float32, [None, 784])
# x = [n,784]



# Weight matrix
W = tf.Variable(tf.zeros([784, 10]))
# 10 neurons in hidden layer

# Bias matrix
b = tf.Variable(tf.zeros([10]))
# Since there

# y = x.w + b
# Softmax activation function
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Loss calculation using cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == "__main__":
    pass