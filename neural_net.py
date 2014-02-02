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
    self.synapses = np.zeros((self.C, self.N, self.D))

  # Using our synapse weights, classify x as one of the C classes
  def classify(self, x):
    if len(x) != self.D: raise Exception("Incorrect length of vector")
    # Classify the vector by taking the dot product with each set of synapses,
    # and picking the class that has the most # of neurons with dot product > theta
    return np.argmax(np.sum(np.dot(self.synapses,x) > self.theta, 1))

  #trainData must be "labeled" - see digit_features.py
  def trainOnSet(self, trainData, numIterations):
    synapses = self.synapses
    T = numIterations
    numTrainPts = len(trainData)

    for t in range(T):
      print t
      # At every sweep through the training points, shuffle the order
      if t % numTrainPts == 0: np.random.shuffle(trainData)
      (x,d) = trainData[t % numTrainPts]

      # We can increment a synapse if it's not maxed out, and vice versa
      canIncrement = self.synapses < 1
      canDecrement = self.synapses > -1

      # Duplicate the input vector C*N times (once per neuron)
      inputIsOn = np.ones((self.C, self.N, self.D)) * x

      # 1 for every synapse of neurons matching the current training point,
      # 0 otherwise
      sameClass = np.zeros((self.C, self.N, self.D))
      sameClass[d] = np.ones((self.N, self.D))

      # A neuron's field is the dot product of synapses times input vector
      fields = np.inner(x, synapses).reshape(self.C, self.N, 1)
      # Must have low field to consider incrementing synapse weights
      lowField = (fields < self.theta + self.delta) 
      lowField = lowField * np.ones((self.C, self.N, self.D))
      # Must have high field to consider decrementing synapse weights
      highField = (fields > self.theta - self.delta) 
      highField = highField * np.ones((self.C, self.N, self.D))

      # Only change synapse weights with probability transP, assuming
      # the other conditions are met
      rands = np.random.random((self.C, self.N, self.D)) < self.transP

      synapsePluses = inputIsOn * sameClass * canIncrement * lowField * rands
      synapseMinuses = inputIsOn * (1 - sameClass) * canDecrement * highField * rands

      # Increment synapses meeting incrementation conditions (see above)
      synapses += synapsePluses
      # Decrement synapses meeting decrementation conditions (see above)
      synapses -= synapseMinuses

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





def testNeuralNet(numIterations):
  trainData = df.labeled(df.flatPixelTrainData())
  testData = df.flatPixelTestData().reshape(10,1000,900)

  net = NeuralNet()
  net.trainOnSet(trainData, numIterations)
  net.testOnSet(testData)
  return net


