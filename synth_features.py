import numpy as np
from numpy.random import random

numTrain = 100  #Number of training points per class
numTest = 1000  #Number of test points per class
P_HIGH = .4
P_LOW = .1
P_BGD = .2 #background pixel probability
N = 100 #length of vectors
M = 10 #number of high/low probability pixels


def posVector():
  highPart = random(M) < P_HIGH
  lowPart = random(M) < P_LOW
  bgPart = random(N - 2*M) < P_BGD
  return 1 * np.append(np.append(highPart, lowPart), bgPart)

def negVector():
  return 1 * (random(N) < P_BGD)

def synthData(ptsPerClass):
  data = np.zeros((2 * ptsPerClass, N));
  for i in range(ptsPerClass):
    data[i] = negVector()
    data[ptsPerClass + i] = posVector()
  return data
    

def labeled(data):
  labeledData = []
  L = len(data)
  for i in range(L):
    labeledData.append((data[i],i*2/L))
  return np.array(labeledData)
