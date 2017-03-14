import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#import seaborn as sns

def standardize (x):
    return (x-x.mean())/x.std()

def square_distance (x, xk):
    return np.linalg.norm(x - xk)**2

def cov_matrix (x):
    n = float(len(x))
    return 1.0/n * x.T.dot(x)

df = pd.read_csv("../Data/arrhythmia.data", header=None)
df = df.replace({"?": np.NaN}).dropna(axis=1, how="any")
df.ix[df.iloc[:, -1] > 1, df.columns[-1]] = 2
df = df.loc[:,(df!=0).any(axis=0)]
df.iloc[:, :-1] = df.iloc[:, :-1].apply(standardize)

def evaluate_performances (prediction, true):
    
    n_strike = float((prediction == true).sum())
    n_miss = float((prediction != true).sum())
    strike_rate = n_strike/(n_strike + n_miss)
    tp = float(((prediction == 2) & (true == 2)).sum())/float((true == 2).sum())
    tn = float(((prediction == 1) & (true == 1)).sum())/float((true == 1).sum())
    fp = float(((prediction == 2) & (true == 1)).sum())/float((true == 2).sum())
    fn = float(((prediction == 1) & (true == 2)).sum())/float((true == 1).sum())
    
    return {
            "strike_rate":strike_rate, 
            "sensitivity":tp, 
            "specificity":tn, 
            "false_positive":fp, 
            "false_negative":fn
           }

def minimum_distance_classification (df):

    xks = df.groupby(df.columns[-1]).mean()    
    y = df.iloc[:, :-1]
    c = df.iloc[:, -1]

    distances = pd.DataFrame(index = df.index, columns = xks.index)
    for class_id in xks.index:
        distances[class_id] = y.apply(square_distance, 
                                      args=(xks.loc[class_id],), 
                                      axis=1)
    prediction = distances.idxmin(axis=1)
    performances = evaluate_performances(prediction, c)
    
    return distances, prediction, performances

dist, min_dist_pred, min_dist_perf = \
    minimum_distance_classification(df)

def PCA (x):
    
    Rx = cov_matrix(x)
    eigvals, U = np.linalg.eig(Rx)
    L = len(np.where(eigvals.cumsum() < eigvals.sum() * 0.999)[0])    

    return U[:, :L-1], L

U, L = PCA(df.iloc[:, :-1])
z = df.iloc[:, :-1].dot(U).apply(standardize)
pca_df = pd.concat([z, df.iloc[:, -1]], axis=1)

dist_pca, min_dist_pca_pred, min_dist_pca_perf= \
    minimum_distance_classification(pca_df)

# Bayes
pi = df.iloc[:, -1].value_counts()/float(len(df))

def bayes_classification_1 (df):

    xks = df.groupby(df.columns[-1]).mean()    
    y = df.iloc[:, :-1]
    c = df.iloc[:, -1]

    distances = pd.DataFrame(index = df.index, columns = xks.index)
    for class_id in xks.index:
        distances[class_id] = y.apply(square_distance, 
                                      args=(xks.loc[class_id],), 
                                      axis=1)
    distances = distances - 2 * np.log(pi)
    prediction = distances.idxmin(axis=1)
    performances = evaluate_performances(prediction, c)
    
    return distances, prediction, performances

dist_bayes_1, bayes_1_pred, bayes_1_perf = \
    bayes_classification_1(pca_df)


def bayes_classification_2 (df):

    xks = df.groupby(df.columns[-1]).mean()    
    y = df.iloc[:, :-1]
    c = df.iloc[:, -1]

    Rxs = df.groupby(df.columns[-1]).apply(cov_matrix)\
                       .drop(279,level=1).drop(279,axis=1) # maybe groupby issue

    distances = pd.DataFrame(index = df.index, columns = xks.index)
    for class_id in xks.index:
        dist = y - xks[class_id]
        Rx = Rxs.loc[class_id]
        Rxi = np.linalg.inv(Rx)
        print dist.shape
        print Rxi.shape
        print dist.T.shape
        print np.log(np.linalg.det(Rx))
        print pi[class_id]
        print ""
        
#    distances = distances - 2 * np.log(pi)
#    prediction = distances.idxmin(axis=1)
#    performances = evaluate_performances(prediction, c)
#    return distances, prediction, performances

bayes_classification_2(pca_df)