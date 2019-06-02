
# coding: utf-8

# In[3]:

import numpy as np
from hmmlearn import hmm
import ast
import json


# In[4]:

class markovmodel:
    #transmat: None
    def __init__(self, transmat = None, startprob = None):
        self.transmat = transmat
        self.startprob = startprob
    # It assumes the state number starts from 0
    def fit(self, X):
        ns = max([max(items) for items in X]) + 1
        self.transmat  = np.zeros([ns, ns])
        self.startprob = np.zeros([ns])
        for items in X:
            n = len(items)
            self.startprob[items[0]] += 1
            for i in range(n-1):
                self.transmat[items[i], items[i+1]] += 1
        self.startprob = self.startprob / sum(self.startprob)
        n = self.transmat.shape[0]
        d = np.sum(self.transmat, axis=1)
        for i in range(n):
            if d[i] == 0:
                self.transmat[i,:] = 1.0 / n
        d[d == 0] = 1
        self.transmat = self.transmat * np.transpose(np.outer(np.ones([ns,1]), 1./d))

    def predict(self, obs, steps):
        pred = []
        n = len(obs)
        if len(obs) > 0:
            s = obs[-1]
        else:
            s = np.argmax(np.random.multinomial(1, self.startprob.tolist(), size = 1))
        for i in range(steps):
            s1 = np.random.multinomial(1, self.transmat[s,:].tolist(), size = 1)
            pred.append(np.argmax(s1))
            s = np.argmax(s1)
        return pred

# In[28]:

def hmm_predict_further_states(ghmm, obs, steps):
    y = ghmm.predict(obs)
    mm = markovmodel(ghmm.transmat_, ghmm.startprob_)
    return mm.predict([y[-1]], steps)

def hmm_predict_future_features(ghmm, obs, steps):
    y = ghmm.predict(obs)
    pred = []
    mm = markovmodel(ghmm.transmat_, ghmm.startprob_)
    sts = mm.predict([], steps)
    for s in sts:
        mean = ghmm.means_[y[-1]]
        cov = ghmm.covars_[y[-1],:]
        x = np.random.multivariate_normal(mean,cov,1)
        pred.append(x[0].tolist())
    return pred

# X: sequence of observations
# y: sequence of latent states
def estimate_parameters(X, y):
    mm = markovmodel()
    mm.fit(y)
    data = dict()
    for i in range(len(y)):
        for s, x in zip(y[i], X[i]):
            if data.has_key(s):
                data[s].append(x)
            else:
                data[s] = [x]
    ns = len(data.keys())
    means = np.array([[np.mean(data[s])] for s in range(ns)])
    covars = np.tile(np.identity(1), (ns, 1, 1))
    for s in range(ns):
        covars[s, 0] = np.std(data[s])
    return mm.startprob, mm.transmat, means, covars


# In[32]:

label = {0: "Seattle", 1: "Boston", 2: "Washington D.C.", 3: "Philadelphia", 4: "New York City"}

X = []
y = []

int_converter = {"Seattle": 0, "Boston": 1, "Washington D.C.": 2, "Philapedia": 3, "New York City": 4}

with open("HW8_Q4_training.txt", "r") as f:
    for line in f.readlines():
        obj = ast.literal_eval(line)
        int_list = []
        cost_list = []
        for element in obj:
            city = element[0]
            cost = element[1]
            city_int = int_converter[city]
            int_list.append(city_int)
            cost_list.append([cost])
        y.append(int_list)
        X.append(cost_list)

# Task 1: Predict the latent cities of training sequences

startprob, transmat, means, covars = estimate_parameters(X, y)
model = hmm.GaussianHMM(5, "full")
print model
model.startprob_ = startprob
model.transmat_ = transmat
model.means_  = means
model.covars_ = covars

for x in X:
    y = model.predict(x)
    print [label[s] for s in y]

file2 = open("HW8_Q4_predictions.txt", "w")

print "Q4:"
with open("HW8_Q4_testing.txt", "r") as f:
    for line in f.readlines():
        element = ast.literal_eval(line)
        X = []
        for integer in element:
            X.append([integer])

        y = model.predict(X)
        print [label[s] for s in y]
        json.dump([label[s] for s in y], file2)
        file2.write("\n")