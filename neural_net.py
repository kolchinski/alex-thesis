import numpy as np
from random import randrange, random
import digit_features as df


class NeuralNet:
  # C is num digits, N num neurons per class, D num features per example
  # theta is field threshold for neurons (>theta is 1, <theta is 0)
  # delta is training threshold: if we aren't at least delta away from the
  # theta threshold for a training point in the correct direction, consider
  # changing the relevant synapse weights to increase separation: bump the
  # synapse weight in the appropriate direction with probability transP
  N = 10
  theta = 0
  delta = 1 
  transP = 0.01
  D = 900
  C = 10

  def __init__(self):
    # Initialize synapse array to correct shape
    self.synapses = np.zeros((self.C,self.N,self.D))

  # Using our synapse weights, classify x as one of the C classes
  def classify(self, x):
    if len(x) != self.D: raise Exception("Incorrect length of vector")
    # Classify the vector by taking the dot product with each set of synapses,
    # and picking the class that has the most # of neurons with dot product > theta
    return np.argmax(np.sum(np.dot(self.synapses,x) > self.theta, 1))

  def trainOnSet(self, trainData):
    synapses = self.synapses
    T = 5000
    for t in range(T):
      print t
      exNum = randrange(len(trainData))
      (x,d) = trainData[exNum]
      
      for c in range(self.C):
        for n in range(self.N):
          totalField = np.dot(x,synapses[c][n])
          for i in range(self.D):
            if x[i] == 1:
              if (d == c) and synapses[c][n][i] < 1 and (totalField < self.theta + self.delta):
                if random() < self.transP: synapses[c][n][i] += 1
              if (d != c) and synapses[c][n][i] > -1 and (totalField > self.theta - self.delta):
                if random() < self.transP: synapses[c][n][i] -= 1

  def testOnSet(self, testData):
    synapses = self.synapses
    totalErrors = 0
    totalTestPoints = 0
    for c in range(self.C):
      numErrors = 0
      numTestPoints = len(testData[c])
      totalTestPoints += numTestPoints
      for i in range(numTestPoints):
        if not self.classify(testData[c][i]) == c:
          numErrors += 1
          totalErrors += 1
      print "Character ", c, ": ", numErrors, " errors out of ", \
          numTestPoints, "; ", 100*numErrors/numTestPoints, " % error"

    print "Total error: ", 100*totalErrors/totalTestPoints, "%"


trainData = df.labeled(df.flatPixelTrainData())
testData = df.flatPixelTestData().reshape(10,1000,900)

net = NeuralNet()
net.trainOnSet(trainData)
net.testOnSet(testData)


##C is num digits, N num neurons per class, D num features per example
#N = 10
#theta = 0
#delta = 1 
#transP = 0.01
#D = 900
#C = 10
#synapses = np.zeros((C,N,D))
#
#
#
#T = 100
#for t in range(T):
#  print t
#  exNum = randrange(len(trainData))
#  (x,d) = trainData[exNum]
#  
#  for c in range(C):
#    for n in range(N):
#      totalField = np.dot(x,synapses[c][n])
#      for i in range(D):
#        if x[i] == 1:
#          if (d == c) and synapses[c][n][i] < 1 and (totalField < theta + delta):
#            if random() < transP: synapses[c][n][i] += 1
#          if (d != c) and synapses[c][n][i] > -1 and (totalField > theta - delta):
#            if random() < transP: synapses[c][n][i] -= 1
#
#
#
#def classify(x, synapses):
#  # Classify the vector by taking the dot product with each set of synapses,
#  # and picking the class that has the most # of neurons with dot product > theta
#  return np.argmax(np.sum(np.dot(synapses,x) > theta, 1))
#  
#
#
#totalErrors = 0
#totalTestPoints = 0
#for c in range(C):
#  numErrors = 0
#  numTestPoints = len(testData[c])
#  totalTestPoints += numTestPoints
#  for i in range(numTestPoints):
#    if not classify(testData[c][i], synapses) == c:
#      numErrors += 1
#      totalErrors += 1
#  print "Character ", c, ": ", numErrors, " errors out of ", numTestPoints, "; ", 100*numErrors/numTestPoints, " % error"
#
#print "Total error: ", 100*totalErrors/totalTestPoints, "%"
#












