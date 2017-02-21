import tensorflow as tf
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
plt.close("all")

# ============================= DATA LOADING =================================
# Data come from MATLAB elaboration, hence have already been properly 
# normalized
dataLoad = scio.loadmat("DataNorm.mat")
dataTrainLoad = scio.loadmat("DataTrainNorm.mat")
dataTestLoad = scio.loadmat("DataTestNorm.mat")

dataNormOriginal = dataLoad.get("DataNorm") 
dataTrainNorm = dataTrainLoad.get("data_train_norm")
dataTestNorm = dataTestLoad.get("data_test_norm")
excludedFeatures = [0, 3, 4, 5, 6]  # Features to be excluded
x_train = np.delete(dataTrainNorm, excludedFeatures, 1)  
nSamplesTrain = len(x_train[:, 0])
nFeatures = len(x_train[0, :])  
regression = [6]     # Features to be regressed
hiddenNodes1 = 17
hiddenNodes2 = 10
flag = True

while flag:
    print("\nNeural Network application\n")
    inp = str(input("Insert a preference\n1 for NO hidden layers\n2 for TWO hidden layes\n3 for delete graphs\n4 Exit\n>>> "))
    
    # ============================ NO HIDDEN NODE CASE ============================
    # =============================================================================
    if inp == "1":
        for i in range(len(regression)):
            y_train = dataTrainNorm[:, regression[i]]   # Regressand feature
            y_train = np.reshape(y_train, (nSamplesTrain, 1))
        
            # ========================= PLACEHOLDERS AND VARIABLES =======================
            # Placeholders are input "container": when application runs the value of the 
            # inputs are overwritten over placeholders. Then, optimization begins cycle by
            # cycle.
            # Initial settings
            tf.set_random_seed(1234)        # in order to get always the same results
            learningRate = 3e-5              # Learning rate for the gradient algorithm    
            xPlaceholder = tf.placeholder(tf.float32,[nSamplesTrain,nFeatures]) 
            yPlaceholder = tf.placeholder(tf.float32, [nSamplesTrain, 1]) # desired output: it is just SPACE
            
            # "Hardware" neural network structure --> NO HIDDEN NODES
            w = tf.Variable(tf.random_normal(shape=[nFeatures, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights"))
            b = tf.Variable(tf.random_normal(shape=[1, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases"))
            y = tf.matmul(xPlaceholder, w) + b      # Activation function --> OUTPUT   
    
            
            # ============================ OPTIMIZATION STRUCTURE =========================
            # Objective function is the reduction of square error
            cost = tf.reduce_sum(tf.squared_difference(y, yPlaceholder, name="objective_function"))
            
            # Use gradient descent in the trainig phase: It is not implemented YET:
            optim = tf.train.GradientDescentOptimizer(learningRate,name="GradientDescent")
            
            # Minimize the objective function changing w and b
            optim_op = optim.minimize(cost, var_list=[w, b])
            
            # Variables initialization
            init=tf.initialize_all_variables()
            #--- run the learning machine
            sess = tf.Session() # Each graph must have its own session
            sess.run(init)
            
            # DUBBIO: DATA GENERATION DOBBIAMO FARLA DENTRO IL FOR????????
            # ============================ GRADIENT ALGORITHM =============================
            for i in range(10000):
                # Data Generation
                xGen = x_train
                yGen = y_train
                yGen = np.reshape(yGen, (nSamplesTrain, 1))
                train_data = {xPlaceholder : xGen, yPlaceholder: yGen}  # Data for feeding placeholder
                sess.run(optim_op, feed_dict=train_data)
            #    if i % 1000 == 0:   # print the intermediate result
            #        print(i,cost.eval(feed_dict=train_data,session=sess))
                    
        #    print("\nFinal results:\nWeights: ")
        #    print(sess.run(w))
        #    print("\nBiases = ")
        #    print(sess.run(b))
#            wHat = sess.run(w)
#            bHat = sess.run(b)
            # Output of the neural network is evaluated:
            yEvaluation = y.eval(feed_dict = train_data, session = sess)
            
            # ========================== RESULT PLOTS ====================================
            plt.figure(), plt.title("NO HIDDEN LAYERS: Regression train for F0 = ")
            plt.plot(y_train, "ro--", label = "Regressand (true)")
            plt.plot(yEvaluation, "bx--", label = "Regression (estimation)")
            plt.grid(which = "major", axis = "both")
            plt.legend(), plt.show()
            
            plt.figure(), plt.title("NO HIDDEN LAYERS: Prediction trend for F0 = ")
            plt.plot(y_train, yEvaluation, "ro")
            plt.xlabel("Regressor: "), plt.ylabel("Regressand: ")
            plt.grid(which = "major", axis = "both")
            plt.legend(), plt.show()
             
            plt.figure(), plt.title("NO HIDDEN LAYERS: Histogram of the errors for F0 = " )
            plt.hist(y_train - yEvaluation, bins = 50)
            plt.grid(which = "major", axis = "both"), plt.show()
            
    # ============================ HIDDEN NODE CASE ===============================
    # =============================================================================        
    elif inp == "2":
        for i in range(len(regression)):
            y_train = dataTrainNorm[:, regression[i]]   # Regressand feature
            y_train = np.reshape(y_train, (nSamplesTrain, 1))        
            tf.set_random_seed(1234)        # in order to get always the same results
            learningRate = 3e-5              # Learning rate for the gradient algorithm    
            xPlaceholder = tf.placeholder(tf.float32,[nSamplesTrain,nFeatures]) 
            yPlaceholder = tf.placeholder(tf.float32, [nSamplesTrain, 1]) # desired output: it is just SPACE
            
            # "Hardware" neural network structure --> There are 17 HIDDEN NODES:
            w1 = tf.Variable(tf.random_normal(shape=[nFeatures, hiddenNodes1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights"))
            b1 = tf.Variable(tf.random_normal(shape=[1, hiddenNodes1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases"))
            a1 = tf.matmul(xPlaceholder, w1) + b1      # Activation function --> OUTPUT   
            z1 = tf.nn.tanh(a1)    # NON-LINEARITY
            w2 = tf.Variable(tf.random_normal(shape=[hiddenNodes1,hiddenNodes2], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
            b2 = tf.Variable(tf.random_normal([1,hiddenNodes2], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
            a2 = tf.matmul(z1, w2) + b2   # neural network output
            z2 = tf.nn.tanh(a2)
            w3 = tf.Variable(tf.random_normal(shape=[hiddenNodes2, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
            b3 = tf.Variable(tf.random_normal(shape=[1,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))        
            y = tf.matmul(z2, w3) + b3        
            cost = tf.reduce_sum(tf.squared_difference(y, yPlaceholder, name="objective_function"))
            optim = tf.train.GradientDescentOptimizer(learningRate,name="GradientDescent")
            optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2, w3, b3])
            init = tf.initialize_all_variables()
            sess = tf.Session() # Each graph must have its own session
            sess.run(init)
    
            # ============================ GRADIENT ALGORITHM =============================
            for k in range(10000):
                # Data Generation
                xGen = x_train
                yGen = y_train
                yGen = np.reshape(yGen, (nSamplesTrain, 1))
                train_data = {xPlaceholder : xGen, yPlaceholder: yGen}  # Data for feeding placeholder
                sess.run(optim_op, feed_dict=train_data)
    #            if i % 1000 == 0:   # print the intermediate result
    #                print(i,cost.eval(feed_dict=train_data,session=sess))
            print(sess.run(w1), sess.run(b1))
            print(sess.run(w2), sess.run(b2))
            print(sess.run(w3), sess.run(b3))
        #    print("\nFinal results:\nWeights: ")
        #    print(sess.run(w))
        #    print("\nBiases = ")
        #    print(sess.run(b))
    #        wHat = sess.run(w)
    #        bHat = sess.run(b)
            # Output of the neural network is evaluated:
            yEvaluation = y.eval(feed_dict = train_data, session = sess)
            
            # ========================== RESULT PLOTS ====================================
            plt.figure(), plt.title("2 HIDDEN LAYERS: Regression train for F0 = ")
            plt.plot(y_train, "ro--", label = "regressand")
            plt.plot(yEvaluation, "bx--", label = "regression")
            plt.legend()
            plt.xlabel("case number")
            plt.grid(which = "major", axis = "both")
            plt.legend(), plt.show()
            
            plt.figure(), plt.title("2 HIDDEN LAYERS: Prediction trend for F0 = ")
            plt.plot(y_train, yEvaluation, "ro")
            plt.xlabel("Regressor: "), plt.ylabel("Regressand: ")
            plt.grid(which = "major", axis = "both")
            plt.legend(), plt.show()
             
            plt.figure(), plt.title("2 HIDDEN LAYERS: Histogram of the errors for F0 = " )
            plt.hist(y_train - yEvaluation, bins = 50)
            plt.grid(which = "major", axis = "both"), plt.show()
    
    elif inp == "3":
        plt.close("all")
        print("Graphs deleted")
        
    elif inp == "4":
        print("Closing...")
        #plt.close("all")
        flag = False
        
# This function is used to restore the initial situazion of the network: it 
# blows away all Variables, tensors and placeholders.
tf.reset_default_graph()