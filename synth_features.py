import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

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

# N is length of vectors, M is number of high/low (as opposed to background)
# probability pixels
def synthData(ptsPerClass, N = 100, M = 10):
  data = np.zeros((2 * ptsPerClass, N));
  for i in range(ptsPerClass):
    data[i] = negVector(N, M)
    data[ptsPerClass + i] = posVector(N, M)
  return data
    
# Decide whether a given vector is positive (1) or negative (0) class,
# assuming we know it was generated using this library
# N is number of high-probability indices = number of low-prob indices
def classifyML(v, N):
  posVectorLik = np.product(np.power(P_HIGH, v[:N])) * \
                 np.product(np.power(1 - P_HIGH, 1 - v[:N])) * \
                 np.product(np.power(P_LOW, v[N:2*N])) * \
                 np.product(np.power(1 - P_LOW, 1 - v[N:2*N]))
  negVectorLik = np.product(np.power(P_BGD, v[:2*N])) * \
                 np.product(np.power(1 - P_BGD, 1 - v[:2*N]))
  return 1 if posVectorLik > negVectorLik else 0


def labeled(data):
  labeledData = []
  L = len(data)
  for i in range(L):
    labeledData.append((data[i],i*2/L))
  return np.array(labeledData)

def plotSynthDataSynapseHistory(synapseHistory):
  posSynapseOnes = [(synapseSet[1] == 1).sum(axis=0)[:100].sum()/10000.0 
      for synapseSet in synapseHistory]
  posSynapseNegs = [(synapseSet[1] == -1).sum(axis=0)[:100].sum()/10000.0 
      for synapseSet in synapseHistory]
  negSynapseOnes = [(synapseSet[1] == 1).sum(axis=0)[100:200].sum()/10000.0 
      for synapseSet in synapseHistory]
  negSynapseNegs = [(synapseSet[1] == -1).sum(axis=0)[100:200].sum()/10000.0 
      for synapseSet in synapseHistory]
  plt.figure(figsize=(12,10))
  plt.plot(posSynapseOnes, label="Proportion weight 1 synapses for high-prob region")
  plt.plot(posSynapseNegs, label="Proportion weight -1 synapses for high-prob region")
  plt.plot(negSynapseOnes, label="Proportion weight 1 synapses for low-prob region")
  plt.plot(negSynapseNegs, label="Proportion weight -1 synapses for low-prob region")
  plt.legend(loc=4)
  plt.show()

