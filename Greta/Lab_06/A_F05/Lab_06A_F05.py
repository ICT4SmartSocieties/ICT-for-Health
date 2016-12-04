#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:23:42 2016

@author: greta
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 08:19:08 2016

@author: greta
""" 
import tensorflow as tf
import numpy as np 
import scipy.io as scio
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
regressand_field = 5-1
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

##--- initial settings
tf.set_random_seed(1234)#in order to get always the same results
Nsamples = rows
learning_rate = 2e-5
x = tf.placeholder(tf.float32,[Nsamples,used_features])#inputs
t = tf.placeholder(tf.float32,[Nsamples,1])#desired outputs
#--- neural netw structure:
w1 = tf.Variable(tf.random_normal(shape=[used_features,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights_1"))
b1 = tf.Variable(tf.random_normal(shape=[Nsamples,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases_1"))
y = tf.matmul(x,w1)+b1

##--- optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim = tf.train.GradientDescentOptimizer(learning_rate,name="GradientDescent")# use gradient descent in the trainig phase
optim_op = optim.minimize(cost, var_list=[w1,b1])# minimize the objective function changing w1,b1,w2,b2

##--- initialize
init = tf.initialize_all_variables()

##--- run the learning machine
sess = tf.Session()
sess.run(init)
for i in range(500000):
    # generate the data
    xval = x_train
    tval = y_train
    # train
    train_data = {x: xval, t: tval}
    sess.run(optim_op, feed_dict = train_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))

##--- graphics
yval=y.eval(feed_dict=train_data,session=sess)


plt.plot(y_train,'r',label='true')
plt.plot(yval,'b',label='estimated')
plt.legend()
plt.title('learning_rate . '+str(learning_rate))
plt.savefig('estimation'+ str(learning_rate)+'.pdf',format='pdf')
plt.show()


hist, bins = np.histogram((y_train-yval), bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlabel('squared error')
plt.title('learning_rate . '+str(learning_rate))
plt.savefig('squared_error'+ str(learning_rate)+'.pdf',format='pdf')
plt.show()



