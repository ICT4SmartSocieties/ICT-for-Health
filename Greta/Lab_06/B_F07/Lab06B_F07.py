#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:27:12 2016

@author: greta
"""
import scipy.io as scio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## ---initialization phase
#importation of matrices
mat_file = scio.loadmat('python_matrix_train.mat')
data_train = mat_file.get('python_matrix_train')
mat_file = scio.loadmat('python_matrix_test.mat') 
data_test = mat_file.get('python_matrix_test')
data_train = np.concatenate((data_train,data_test),axis=0)

#initialization of useful variables
features = len(data_train[1])
rows = len(data_train)
patient_field = 1-1
time_field = 4-1
regressand_field = 7-1
motor_UPDRS_field = 5-1
total_UPDRS_field = 6-1
not_used_fields = ([[patient_field,time_field,motor_UPDRS_field,total_UPDRS_field,regressand_field]])
#normalization
mean = np.mean(data_train,0)
std = np.std(data_train,0)
variance = np.var(data_train,0)
for i in range(0,rows):
    data_train[i] = (data_train[i]-mean)/std

#preparation of data for regression
x_train = np.delete(data_train,not_used_fields,1)    #not to consider useless features
used_features = len(x_train[1])
y_train = data_train[:,regressand_field:regressand_field+1]
#--- initial settings
tf.set_random_seed(1234)#in order to get always the same results
Nsamples = rows
hidden_nodes_first_layer = 17
hidden_nodes_second_layer = 10
learning_rate = 2e-6
x = tf.placeholder(tf.float32,[Nsamples,used_features])#inputs
t = tf.placeholder(tf.float32,[Nsamples,1])#desired outputs
#--- neural netw structure:
w1 = tf.Variable(tf.random_normal(shape=[used_features,hidden_nodes_first_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
b1 = tf.Variable(tf.random_normal(shape=[1,hidden_nodes_first_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))
a1 = tf.matmul(x,w1)+b1
z1 = tf.nn.tanh(a1)
w2 = tf.Variable(tf.random_normal([hidden_nodes_first_layer,hidden_nodes_second_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2 = tf.Variable(tf.random_normal([1,hidden_nodes_second_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
a2 = tf.matmul(z1,w2)+b2
z2 = tf.nn.tanh(a2)
w3 = tf.Variable(tf.random_normal([hidden_nodes_second_layer,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
b3 = tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))
y = tf.matmul(z2,w3)+b3
#--- optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim = tf.train.GradientDescentOptimizer(learning_rate,name="GradientDescent")# use gradient descent in the trainig phase
optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2,w2,b3])# minimize the objective function changing w1,b1,w2,b2
#--- initialize
init = tf.initialize_all_variables()
#--- run the learning machine
sess = tf.Session()
sess.run(init)
for i in range(50000):
    # generate the data
    xval = x_train
    tval = y_train
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
#    if i % 100 == 0:# print the intermediate result
#        print(i,cost.eval(feed_dict=train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))
print(sess.run(w2),sess.run(b2))
print(sess.run(w3),sess.run(b3))

##---graphics
yval=y.eval(feed_dict=train_data,session=sess)

plt.plot(y_train,'r',label='true')
plt.plot(yval,'b',label='estimated')
plt.legend()
plt.title('leaning rate = ' +str(learning_rate))
plt.savefig('estimation'+str(learning_rate)+'.pdf',format='pdf')
plt.show()

hist, bins = np.histogram((y_train-yval), bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlabel('squared error')
plt.title('leaning rate = ' +str(learning_rate))
plt.savefig('squared_error'+str(learning_rate)+'.pdf',format='pdf')
plt.show()
