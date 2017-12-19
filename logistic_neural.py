# Neural network
# In the neural network terminology:
#
# one epoch = one forward pass and one backward pass of all the training examples

# batch size = the number of training examples in one forward/backward pass.
# The higher the batch size, the more memory space you'll need.

# number of iterations = number of passes, each pass using [batch size]
# number of examples. To be clear, one pass = one forward pass + one backward pass
# (we do not count the forward pass and backward pass as two different passes).

# Example: if you have 1000 training examples, and your batch size is 500, then
# it will take 2 iterations to complete 1 epoch.

# Fully connected Neural Net ==> Multilayer perceptron
# With 2 Hidden layers

# Again with MNIST data

import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle


# Training Param
learning_rate = 0.01
batch_size = 128    # Batch size defines number of samples that going to be propagated through the network.
num_step = 1000      # Epochs
display_step = 100  # Show loss after every steps

# Network param
n_hidden1 = 128     # neurons in first hidden layer
n_hidden2 = 128    # neurons in second hidden layer e
                    # ach image is 28 pixels by 28 pixels = 28*28 = 784 input neurons
n_input = 9      # neurons in input layer
num_classes = 1    # MNIST total classes (0-9 digits) # Depends type of data set that you have taken



X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, num_classes])

weight_h1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
weight_h2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
weight_out = tf.Variable(tf.random_normal([n_hidden2, num_classes]))

bias_h1 = tf.Variable(tf.random_normal([n_hidden1]))
bias_h2 = tf.Variable(tf.random_normal([n_hidden2]))
bias_out = tf.Variable(tf.random_normal([num_classes]))


def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weight_h1), bias_h1)
    layer_2 = tf.add(tf.matmul(layer_1, weight_h2), bias_h2)
    # Output fully connected layer with a neuron for each class
    layer_out = tf.add(tf.matmul(layer_2, weight_out), bias_out)
    return layer_out

# Construct model
logits = neural_net(X)
prediction = tf.nn.sigmoid(logits)

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels= Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(prediction, Y) # Check for correct results

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # nothing but average kind of thing

init = tf.global_variables_initializer()


# load training data
data = pd.read_csv("./data/logistic_regression_data_3.csv")
# shuffled_data = shuffle(data)
# train_x = shuffled_data["i"].values.reshape(100,1)
# train_y = shuffled_data["o"].values.reshape(100,1)
# print(shuffled_data.iloc[:,1:2].shape())
train_x = data.iloc[:, :9]
train_y = data.iloc[:, 9:]


with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_step+1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict= {X:train_x, Y:train_y})
        if step % display_step == 0 or step == 1:
            # print(sess.run(prediction, feed_dict={X: train_x, Y: train_y}))
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_x,
                                                                Y: train_y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    # Calculate accuracy for MNIST test images
    # print("Testing Accuracy:", \
    #       sess.run(accuracy, feed_dict={X: mnist.test.images,
    #                                     Y: mnist.test.labels}))

    # for x in train_x:
    #     x = x.reshape(1,1)
    print(sess.run(prediction, feed_dict={X:train_x}))
    res = sess.run(prediction, feed_dict={X:train_x})
    res = pd.DataFrame(res)
    train_x =  pd.DataFrame(train_x)
    train_x.to_csv("odd_even_data.csv")
    res.to_csv("odd_even_result.csv")
