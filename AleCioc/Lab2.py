import numpy as np
import pandas as pd

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


def lab2():
    
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
    print len(train_df)
    
    test_df = new_df[new_df.subject >= 38]
    test_df.to_csv("testing_data.csv")
    print len(test_df)
    
    training_data = pd.DataFrame()
    testing_data = pd.DataFrame()
    for col in train_df:
        if col not in ["day","age","sex","subject","test_time"]:
            training_data[col] = (train_df[col].values-train_df[col].mean())/train_df[col].std()
            testing_data[col] = (test_df[col].values-test_df[col].mean())/test_df[col].std()    
            #print training_data[col].mean(), training_data[col].var()
            #print testing_data[col].mean(), testing_data[col].var()
            
    #x_col = "Shimmer"
    y_col = "Jitter(%)"
    
    y_train = training_data[y_col].astype("float64").values
    X_train = training_data[[col for col in training_data if col != y_col]] \
                   .astype("float64").values

    y_test = testing_data[y_col].astype("float64").values
    X_test = testing_data[[col for col in testing_data if col != y_col]] \
                   .astype("float64").values

    #print training_data.var()

    def PCR(X, y, N):

        RX = 1.0/N * X.T.dot(X)
        print RX
        eigvals, U = np.linalg.eig(RX)
        
        for i in range(len(eigvals)):
            print eigvals[i]
                
        Lambda = np.diag(eigvals)
        a = 1.0/N * U.dot(np.linalg.inv(Lambda).dot(U.T).dot(X.T).dot(y))        
        return a

    a = PCR(X_train, y_train, len(training_data)-1)
    yhat_train = X_train.dot(a)
    yhat_test =  X_test.dot(a)
    plot_LR_performances(yhat_train, y_train, yhat_test, y_test)
        
lab2()