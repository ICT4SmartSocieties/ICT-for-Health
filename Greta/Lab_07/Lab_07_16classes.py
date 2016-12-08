#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:32:59 2016

@author: greta
"""

import tensorflow as tf
import numpy as np 
import scipy.io as scio
#import matplotlib.pyplot as plt

## ---initialization phase
#importation of matrix
mat_file = scio.loadmat('arrhythmia.mat')
data = mat_file.get('arrhythmia')
file = open("results_16classes", "a")
file.write("\n")
file.write("*************************************\n")

#initialization of useful variables
(patients,features) = np.shape(data)
class_field = features-1
classes = 16
estimated_class = np.zeros(patients)
patients_per_classes = np.zeros(classes)
pi = np.zeros(classes)

#elimination of columns with all zeros
columns_to_eliminate =[]
for i in range(0,features):
    if np.shape(np.nonzero(data[:,i]))[1] == 0:
        columns_to_eliminate.append(i)

data = np.delete(data,columns_to_eliminate,1)
(patients,features) = np.shape(data)

# x_train and  y(class_id) definition
x_train = data[:,0:features-1]
class_id_vector = data[:,features-1:features]-1
used_features = np.shape(x_train)[1]

#since we have 16 classes, we use a matrix, one row for each patient, column = 1 if the column corresponds to the class
class_id_matrix = np.zeros([patients,classes])
for i in range (0,patients):
    class_id_matrix[i][int(class_id_vector[i])] = 1
    patients_per_classes[int(class_id_vector[i])] +=1 #count of patients per class
    
#computation of probability to be in a certain class
pi = patients_per_classes/patients

#normalization
mean = np.mean(x_train,0)
std = np.std(x_train,0)
variance = np.var(x_train,0)
for i in range(0,patients):
    x_train[i] = (x_train[i]-mean)/std

#--- initial settings
iterations = 10000
learning_rate = 2e-6
tf.set_random_seed(1234)#in order to get always the same results
Nsamples = patients
outputs = classes
hidden_nodes_first_layer = 64
hidden_nodes_second_layer = 32
x = tf.placeholder(tf.float32,[Nsamples,used_features])#inputs
t = tf.placeholder(tf.float32,[Nsamples,outputs])#desired outputs
#--- neural netw structure:
w1 = tf.Variable(tf.random_normal(shape=[used_features,hidden_nodes_first_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
b1 = tf.Variable(tf.random_normal(shape=[1,hidden_nodes_first_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))
a1 = tf.matmul(x,w1)+b1
z1 = tf.nn.sigmoid(a1)
w2 = tf.Variable(tf.random_normal([hidden_nodes_first_layer,hidden_nodes_second_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2 = tf.Variable(tf.random_normal([1,hidden_nodes_second_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
a2 = tf.matmul(z1,w2)+b2
z2 = tf.nn.sigmoid(a2)
w3 = tf.Variable(tf.random_normal([hidden_nodes_second_layer,outputs], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
b3 = tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))
a3 = tf.matmul(z2,w3)+b3
y = tf.nn.softmax(a3) #output of the network 
#--- optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim = tf.train.GradientDescentOptimizer(learning_rate,name="GradientDescent")# use gradient descent in the trainig phase
optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2,w3,b3])# minimize the objective function changing w1,b1,w2,b2
#--- initialize
init = tf.initialize_all_variables()
#--- run the learning machine
sess = tf.Session()
sess.run(init)
for i in range(iterations):
    # generate the data
    xval = x_train
    tval = class_id_matrix
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict = train_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict = train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))
print(sess.run(w2),sess.run(b2))
print(sess.run(w3),sess.run(b3))

#computation of statistics
correct_decisions = np.zeros(classes)
probability_correct_decisions = 0
yval = y.eval(feed_dict=train_data,session=sess)
for i in range (0,patients):
    if np.argmax(yval[i]) == int(class_id_vector[i]):
        correct_decisions[int(class_id_vector[i])] +=1
        
for i in range (0,classes):
    if patients_per_classes[i] > 0:
        correct_decisions[i] = correct_decisions[i]/patients_per_classes[i]
        probability_correct_decisions += correct_decisions[i]*pi[i]

print(str(probability_correct_decisions))
file.write("learning rate : "+str(learning_rate)+"    "+"iterations : "+str(iterations)+"\n")
file.write("probability correct decision: "+str(probability_correct_decisions)+"\n2")
file.close()
