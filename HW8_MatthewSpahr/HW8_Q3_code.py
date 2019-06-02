
# coding: utf-8

# In[2]:

import numpy as np
import types
import itertools
import copy
import ast
import json


# In[23]:

def get_all_sequences(m, n):
    i = 1
    S = []
    for j in range(n):
        S.append([j])
    while i < m:
        S1 = []
        for s in S:
            for j in range(n):
                s1 = copy.deepcopy(s)
                s1.append(j)
                S1.append(s1)
        S.extend(S1)
        i = i + 1
    S = [item for item in S if len(item) == m]
    return S


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

#    def predict_most_probable_sequene(self, ):


    def predict(self, obs, steps):
        n = len(obs)
        if len(obs) > 0:
            combs = get_all_sequences(steps, len(self.startprob))
            max_seq = []
            max_prob = -1
            for comb in combs:
                prob = 1.0
                prev = obs[-1]
                for i in comb:
                    prob = prob * self.transmat[prev, i]
                    prev = i
                if prob > max_prob:
                    max_seq = comb
                    max_prob = prob
            print max_prob,max_seq
            return max_seq
        else:
            combs = get_all_sequences(steps, len(self.startprob))
            max_seq = []
            max_prob = -1
            for comb in combs:
                prob = 1.0
                prev = -1
                for i in comb:
                    if prev == -1:
                        prob = prob * self.startprob[i]
                    else:
                        prob = prob * self.transmat[prev, i]
                    prev = i
                if prob > max_prob:
                    max_seq = comb
                    max_prob = prob
            return max_seq


# In[1]:

label = {0: "Seattle", 1: "Boston", 2: "Washington D.C.", 3: "Philapedia", 4: "New York City"}
int_converter = {"Seattle": 0, "Boston": 1, "Washington D.C.": 2, "Philapedia": 3, "New York City": 4}
y = []
with open("HW8_Q3_training.txt", "r") as file:
    for line in file.readlines():
        obj = ast.literal_eval(line)
        int_list = []
        for city in obj:
            city_int = int_converter[city]
            int_list.append(city_int)
        y.append(int_list)

# train a markov model
mm = markovmodel()
mm.fit(y)

file2 = open("HW8_Q3_predictions.txt", "w")

print "Q3:"
with open("HW8_Q3_testing.txt", "r") as file3:
    for line in file3.readlines():
        obj = ast.literal_eval(line)
        pred_list = []
        for city in obj:
            pred_int = int_converter[city]
            pred_list.append(pred_int)
        pred = mm.predict(pred_list, 5)
        print [label[s] for s in pred]
        json.dump([label[s] for s in pred], file2)
        file2.write("\n")
file2.close()