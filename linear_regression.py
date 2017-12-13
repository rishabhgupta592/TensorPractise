
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

import numpy as np

learning_rate = 0.01
step = 3000

data = pd.read_csv("./data/linear_regression_data.csv")
train_x = data['I']
train_y = data['O']

train_x = preprocessing.normalize([train_x], norm='l2')
# train_x = train_x.reshape(1,1)
# need to scale data
# print(train_x)
train_x = train_x.reshape(100,1)
# print(train_x)
# train_x = np.linspace(-1, 1, 101)
# train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.33

X = tf.placeholder("float")  # actual input variable
Y = tf.placeholder("float") # actual output variable

w = tf.Variable(10.0, name="weights")
b = tf.Variable(5.0, name="bias")

# y = wx+b
y_model = tf.add(tf.multiply(X,w),b)
# y_model = tf.multiply(X,w)
# Root mean square error
cost = tf.square(Y - y_model)

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch grpah
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(step):
        print("Weight -> ", i, sess.run(w))
        for(x,y) in zip(train_x, train_y):
            # print("Cost --> ", sess.run(cost, feed_dict={X: x, Y: y}))
            sess.run(train_op, feed_dict={X:x, Y:y})

    print(sess.run(w))
    print(sess.run(b))

    for(x,y) in zip(train_x, train_y):
        print(sess.run(y_model, feed_dict={X:x}))