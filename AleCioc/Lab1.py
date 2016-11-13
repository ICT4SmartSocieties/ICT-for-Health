import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

def plot_LR_performances (yhat_train, y_train, yhat_test, y_test):

    fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize = (14,6))
    ax1.plot(yhat_train, label="Data")
    ax1.plot(y_train, label="Data")
    sns.distplot(yhat_train-y_train, label="Data", ax=ax2)
    
    ax3.plot(yhat_test, label="Data")
    ax3.plot(y_test, label="Data")
    sns.distplot(yhat_test-y_test, label="Data", ax=ax4)
    #plt.show()

def lab1 ():

    df = pd.read_csv("parkinsons_updrs.csv")
    	
    # Time
    time = df["test_time"]
    for idx in time[time < 0].apply(np.abs).index:
    	df.set_value(idx, "test_time", time[time < 0].apply(np.abs).loc[idx])
    df["day"] = df["test_time"].astype("int64")
    
    # Pre-processing
    new_df = pd.DataFrame(columns=df.columns)
    i = 0
    for subject, measurements in df.groupby(["subject"]):		
    	for day, day_measurements in measurements.groupby("day"):
    		new_df.set_value(i, "subject", subject)
    		for col in day_measurements:
    			if col not in ["subject"]:
    				new_df.set_value(i, col, day_measurements[col].mean())					
    		i += 1
    
    # Define training and testing
    train_df = new_df[new_df.subject < 38]
    train_df.to_csv("training_data.csv")
    
    test_df = new_df[new_df.subject >= 38]
    test_df.to_csv("testing_data.csv")

    training_data = pd.DataFrame()
    testing_data = pd.DataFrame()
    for col in train_df:
        if col not in ["day","age","sex","subject","test_time"]:
            training_data[col] = (train_df[col].values-train_df[col].mean())/train_df[col].std()
            testing_data[col] = (test_df[col].values-test_df[col].mean())/test_df[col].std()    
            #print training_data[col].mean(), training_data[col].var()
            
    training_data.to_csv("training_norm_data.csv")
    testing_data.to_csv("testing_norm_data.csv")
    
    #x_col = "Shimmer"
    y_col = "Jitter(%)"
    #plot_SLR (testing_data, x_col, y_col)
    
    # Let's train
    y = training_data[y_col].astype("float64").values
    X = training_data[[col for col in training_data if col != y_col]] \
                   .astype("float64").values
    X_test = testing_data[[col for col in training_data if col != y_col]] \
                   .astype("float64").values
    y_test = testing_data[y_col].astype("float64").values

    print X.shape
    print y.shape

    def MLS_LR_smgg():
        pass            
    def MLS_LR_slides():
        pass
    def GD_LR_slides():
        pass
    def GD_LR_web():
        pass
    def STEEPD_LR_slides():
        pass
    def STOCHD_LR_slides():
        pass
    
    # OLSLR by statsmodels
    sm_ols = sm.OLS(y, X).fit()
    print(sm_ols.summary())
    print sm_ols.params
    
    # MLSLR (slides version)
    weights = np.dot(np.linalg.pinv(X),y)
    loss = X.dot(weights) - y
    print np.sum(loss ** 2)
    print weights
    
    y_train = y
    yhat_test = X_test.dot(weights)
    yhat_train = X.dot(weights)    
    plot_LR_performances(yhat_train, y_train, yhat_test, y_test)
    
    # Gradient descent LR (web version)
    learning_coefficient = 0.2
    a = np.array([train_df[col].mean() for col in train_df if col not in \
    					["day","age","sex","subject","test_time",y_col]])
    for i in range(0, 100000):
        yhat = X.dot(weights)
        loss = yhat - y
        cost = np.sum(loss ** 2)
        gradient = X.T.dot(loss) / 2.0 / float(len(y))
        a -= learning_coefficient * gradient
    print cost
    print a
    y_train = y
    yhat_test = X_test.dot(a)
    yhat_train = X.dot(a)    
    plot_LR_performances(yhat_train, y_train, yhat_test, y_test)

    # Gradient descent LR (slides version)
    learning_coefficient = 0.00001
    weights = np.array([train_df[col].mean() for col in train_df if col not in \
    					["day","age","sex","subject","test_time",y_col]])
    for i in range(0, 100000):
        yhat = X.dot(weights)
        loss = yhat - y
        cost = np.sum(loss ** 2)
        gradient = -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(weights)
        a -= learning_coefficient * gradient
    print cost
    print a
    y_train = y
    yhat_test = X_test.dot(a)
    yhat_train = X.dot(a)    
    plot_LR_performances(yhat_train, y_train, yhat_test, y_test)

    # Steepest descent LR (slides version)
    a = np.array([train_df[col].mean() for col in train_df if col not in \
    					["day","age","sex","subject","test_time",y_col]])
    for i in range(0, 100000):
        yhat = X.dot(weights)
        loss = yhat - y
        cost = np.sum(loss ** 2)
        gradient = -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)
        H = 4 * X.T.dot(X)
        a -= np.dot((gradient.T.dot(gradient) / (gradient.T.dot(H).dot(gradient))),gradient)
    print cost
    print a
    y_train = y
    yhat_test = X_test.dot(a)
    yhat_train = X.dot(a)    
    plot_LR_performances(yhat_train, y_train, yhat_test, y_test)
    
    fig,ax = plt.subplots()
    ax.plot(range(0, len(a)), a, "o")

    plt.show()

lab1()