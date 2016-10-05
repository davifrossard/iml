
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# *Load necessary libraries*

# In[2]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# *Load MNIST Dataset*

# In[3]:

init_param = lambda shape: tf.random_normal(shape, dtype=tf.float32)

with tf.name_scope("IO"):
    inputs = tf.placeholder(tf.float32, [None, 784], name="X")
    targets = tf.placeholder(tf.float32, [None, 10], name="Yhat")

with tf.name_scope("LogReg"):
    W = tf.Variable(init_param([784, 10]), name="W")
    B = tf.Variable(init_param([10]))
    logits = tf.matmul(inputs, W) + B
    y = tf.nn.softmax(logits)
    
with tf.name_scope("train"):
    learning_rate = tf.Variable(0.5, trainable=False)
    cost_op = tf.nn.softmax_cross_entropy_with_logits(logits, targets)
    cost_op = tf.reduce_mean(cost_op) 
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(targets,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100


# *Create TensorFlow graph*

# In[4]:

tolerance = 1e-4
# Perform Stochastic Gradient Descent
epochs = 1
last_cost = 0
alpha = 0.7
max_epochs = 100
batch_size = 50
costs = []
sess = tf.Session()
print "Beginning Training"
with sess.as_default():
    init = tf.initialize_all_variables()
    sess.run(init)
    sess.run(tf.assign(learning_rate, alpha))
    writer = tf.train.SummaryWriter("/tmp/tboard", sess.graph) # Create TensorBoard files
    while True:
        
        num_batches = int(mnist.train.num_examples/batch_size)
        cost=0
        for _ in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            tcost, _ = sess.run([cost_op, train_op], feed_dict={inputs: batch_xs, targets: batch_ys})
            cost += tcost
        cost /= num_batches

        tcost = sess.run(cost_op, feed_dict={inputs: mnist.test.images, targets: mnist.test.labels})
            
        costs.append([cost, tcost])
        
        # Keep track of our performance
        if epochs%5==0:
            acc = sess.run(accuracy, feed_dict={inputs: mnist.train.images, targets: mnist.train.labels})
            print "Epoch: %d - Error: %.4f - Accuracy - %.2f%%" %(epochs, cost, acc)

            # Stopping Condition
            if abs(last_cost - cost) < tolerance or epochs > max_epochs:
                print "Converged."
                break

            last_cost = cost
            
        epochs += 1
    
    tcost, taccuracy = sess.run([cost_op, accuracy], feed_dict={inputs: mnist.test.images, targets: mnist.test.labels})
    print "Test Cost: %.4f - Accuracy: %.2f%% " %(tcost, taccuracy)


# *Perform gradient descent to learn model*

# In[5]:

epochs = len(costs)
costs = np.array(costs)
plt.plot(range(epochs), costs[:,0], label="Training")
plt.plot(range(epochs), costs[:,1], label="Test")
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy")
plt.title("Training Curve")
plt.legend(loc='best')
plt.show()


# *Plot train curves*
