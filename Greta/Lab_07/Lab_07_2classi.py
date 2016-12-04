#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:22:52 2016

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
#file = open("results_2classes", "a")
#file.write("\n")
#file.write("*************************************\n")

#initialization of useful variables
patients_status2 = 0
patients_status1 = 0
classes = 2
(patients,features) = np.shape(data)
class_field = features-1
# put all classes>1 to 2
for i in range (0,patients):
    if data[i][class_field] > 1:
        data[i][class_field] = 2
        patients_status2 += 1
    else:
        patients_status1 +=1

#elimination of columns with all zeros
columns_to_eliminate =[]
for i in range(0,features):
    if np.shape(np.nonzero(data[:,i]))[1] == 0:
        columns_to_eliminate.append(i)

data = np.delete(data,columns_to_eliminate,1)
(patients,features) = np.shape(data)

# x e y definition
x_train = data[:,0:features-1]
class_id = data[:,features-1:features]-1
used_features = np.shape(x_train)[1]

#normalization
mean = np.mean(x_train,0)
std = np.std(x_train,0)
variance = np.var(x_train,0)
for i in range(0,patients):
    x_train[i] = (x_train[i]-mean)/std


#--- initial settings
iterations = 1000
outputs = 1
learning_rate = 2e-2
tf.set_random_seed(1234)#in order to get always the same results
Nsamples = patients
hidden_nodes_first_layer = 257
hidden_nodes_second_layer = 128
x = tf.placeholder(tf.float32,[Nsamples,used_features])#inputs
t = tf.placeholder(tf.float32,[Nsamples,1])#desired outputs
#--- neural netw structure:
w1 = tf.Variable(tf.random_normal(shape=[used_features,hidden_nodes_first_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
b1 = tf.Variable(tf.random_normal(shape=[Nsamples,hidden_nodes_first_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))
a1 = tf.matmul(x,w1)+b1
z1 = tf.nn.sigmoid(a1)
w2 = tf.Variable(tf.random_normal([hidden_nodes_first_layer,hidden_nodes_second_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2 = tf.Variable(tf.random_normal([Nsamples,hidden_nodes_second_layer], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
a2 = tf.matmul(z1,w2)+b2
z2 = tf.nn.sigmoid(a2)
w3 = tf.Variable(tf.random_normal([hidden_nodes_second_layer,outputs], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
b3 = tf.Variable(tf.random_normal([Nsamples,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))
a3 = tf.matmul(z2,w3)+b3
y = tf.nn.sigmoid(a3) #output of the network 
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
    tval = class_id
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict = train_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict = train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))
print(sess.run(w2),sess.run(b2))
print(sess.run(w3),sess.run(b3))
yval = y.eval(feed_dict=train_data,session=sess)
yval = np.round(yval) #because the output is a probability: tval is the probability to belonge to class 1(2 in the original file)-->prob to have the desease

#--statistics evaluations
true_positive = 0
false_negative =0
true_negative = 0
false_positive =0

for i in range(0,patients):
    if (yval[i]) == 1 and class_id[i,0] ==1.0:
        true_positive+=1
    if (yval[i]) == 1 and class_id[i,0] ==0:
        false_positive+=1
    if (yval[i]) == 0 and class_id[i,0] ==0:
        true_negative+=1
    if (yval[i]) == 0 and class_id[i,0] ==1.0:
        false_negative+=1
#patients with desease  
probability_true_positive = true_positive/patients_status2
probability_false_negative = false_negative/patients_status2
#patient without desease
probability_false_positive = false_positive/patients_status1
probability_true_negative = true_negative/patients_status1
## print of output
print("probability_true_positive : "+str(probability_true_positive)+"\n")
print("probability_false_negative : "+str(probability_false_negative)+"\n")
print("probability_false_positive : "+str(probability_false_positive)+"\n")
print("probability_true_negative : "+str(probability_true_negative)+"\n2")

#file.write("learning rate : "+str(learning_rate)+"    "+"iterations : "+str(iterations)+"\n")
#file.write("probability_true_positive : "+str(probability_true_positive)+"\n")
#file.write("probability_false_negative : "+str(probability_false_negative)+"\n")
#file.write("probability_false_positive : "+str(probability_false_positive)+"\n")
#file.write("probability_true_negative : "+str(probability_true_negative)+"\n2")
#file.close()