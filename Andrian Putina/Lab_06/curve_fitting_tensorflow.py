import tensorflow as tf
import numpy as np

import scipy.io as scio

mat_file = scio.loadmat('finalMatrix.mat')
data = mat_file.get('finalMatrix')

data=np.asmatrix(data)

print data

ytest = data[:,4]

print "-----------"
print ytest
data = np.delete(data, [0,3,4], 1) 

mean = np.mean(data,0)
std = np.std(data,0)

data = data - mean
data = data / std



rig,col=data.shape
print "-----------"

#--- initial settings
tf.set_random_seed(1234)#in order to get always the same results
Nsamples=rig
x=tf.placeholder(tf.float32,[Nsamples,col])#inputs
t=tf.placeholder(tf.float32,[Nsamples,1])
z=tf.placeholder(tf.float32,[Nsamples,1])#desired outputs
#--- neural netw structure:
w1=tf.Variable(tf.random_normal(shape=[col,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights"))
b1=tf.Variable(tf.random_normal(shape=[1,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases"))
a1=tf.matmul(x,w1)+b1
y=a1

#--- optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim=tf.train.GradientDescentOptimizer(2e-5,name="GradientDescent")# use gradient descent in the trainig phase
optim_op = optim.minimize(cost, var_list=[w1,b1])# minimize the objective function changing w1,b1,w2,b2
#--- initialize
init=tf.initialize_all_variables()
#--- run the learning machine
sess = tf.Session()
sess.run(init)
xval=data
tval=ytest

for i in range(10000):
    
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 100 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))