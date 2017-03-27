from scipy.io.wavfile import read
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

z = pd.DataFrame()
for vowel in ["a","e","i","o","u"]:
    for i in range(1,6):
        z[vowel+str(i)] = pd.Series(read(vowel + str(i) + ".wav")[1])

M = 8
N = 5

def _cut (s):
    s = s.loc[s > 130]
    s = s.reset_index(drop=True).fillna(np.mean)
    return s.iloc[:200]

z = z.apply(_cut).dropna().values.ravel()

from hmmlearn import hmm
np.random.seed(42)

print "fitting to HMM and decoding ..."
# Make an HMM instance and execute fit
model = hmm.GaussianHMM(n_components=5, 
                    covariance_type="full", 
                    n_iter=1000).fit(z.reshape(-1,1))

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(z.reshape(-1,1))

print("done")