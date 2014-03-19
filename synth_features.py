import numpy as np
from numpy.random import random

numTrain = 100  #Number of training points per class
numTest = 1000  #Number of test points per class
P_HIGH = .4
P_LOW = .1
P_BGD = .2 #background pixel probability

def posVector(N = 100, M = 10):
  highPart = random(M) < P_HIGH
  lowPart = random(M) < P_LOW
  bgPart = random(N - 2*M) < P_BGD
  return 1 * np.append(np.append(highPart, lowPart), bgPart)

def negVector(N = 100, M = 10):
  return 1 * (random(N) < P_BGD)

# M is length of vectors, N is number of high/low (as opposed to background)
# probability pixels
def synthData(ptsPerClass, N = 100, M = 10):
  data = np.zeros((2 * ptsPerClass, N));
  for i in range(ptsPerClass):
    data[i] = negVector(N, M)
    data[ptsPerClass + i] = posVector(N, M)
  return data
    

def labeled(data):
  labeledData = []
  L = len(data)
  for i in range(L):
    labeledData.append((data[i],i*2/L))
  return np.array(labeledData)
