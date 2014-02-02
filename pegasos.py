import numpy as np
from scipy import misc
from random import randrange, getrandbits, choice,sample
import matplotlib.pyplot as plt
import digit_features as df

unlabeledTrainData = df.flatEdgeTrainData()
unlabeledTestData = df.flatEdgeTestData()
trainData = df.labeled(unlabeledTrainData)
testData = df.labeled(unlabeledTestData)

D = trainData[0][0].size
C = 10

W = np.zeros((C,D))

def decideIfDigit(x,d):
  totalVal = np.dot(W[d],x)
  if totalVal > 100: return True
  else: return False

def classify(x):
  dots = [np.dot(W[d], x) for d in range(C)]
  return dots.index(max(dots))

# S is num samples, T num iterations, l learning rate
S = len(trainData)
T = 8000
#l = 0.01
l = 0.5
k = 1

#Train one-vs-the-rest classifiers one by one
#This could be vectorized, but it runs quickly as it is
for c in range(C):
  print "\nTraining one-vs-rest SVM for class {}".format(c)
  for t in range(1, T + 1):
    if t % (T/10) == 0: 
      weightPenalty = l / 2.0 * np.linalg.norm(W[c])**2
      dotProducts = np.inner(unlabeledTrainData, W[c])
      #Ys is 1 if class matches, -1 if not
      Ys = (np.arange(S) / (S/10) == c) * 2 - 1
      losses = (1 - Ys * dotProducts)
      positiveLosses = (losses > 0) * losses
      avgLoss = np.mean(positiveLosses)
      totalLoss = weightPenalty + avgLoss
      print "Total loss: {}".format(totalLoss)

    AsubT = trainData[sample(range(S),k)]
    #etaT = 1.0 / l / (t*5 + 1)
    etaT = 1.0 / l / (t + 1)
    W[c] *= 1.0 * (t - 1) / t
    for i in range(k):
      (x,d) = AsubT[i]
      y = 1 if (d == c) else -1
      if np.dot(W[c], x) * y < 1:
        W[c] += etaT / k * y * x
    
    wNorm = np.linalg.norm(W[c])
    maxNorm = 1.0 / np.sqrt(l)
    if wNorm > maxNorm: 
      W[c] *= (maxNorm / wNorm)
      

classifications = np.zeros((10,10))

totalErrors = 0
numTestPoints = len(testData)
for i in range(numTestPoints):
  x,c = testData[i]
  classifications[c,classify(x)] += 1
  if not classify(x) == c:
    totalErrors += 1

print "Total error: ", 100*totalErrors/numTestPoints, "%"
for c in range(C):
  rights = classifications[c][c]
  total = np.sum(classifications[c])
  print "Error for class {}: {}%".format(c, 100.0*(total - rights)/total)
print classifications


#Graph learned digit representations for edges
#M = D/8
#plt.close('all')
#meanEdges = np.mean(W.reshape(10,30,30,8), axis=3)
#for i in range(C):
#  w = W[i].reshape((30,30,8))
#  #w = df.edgeTrainData()[i*100+1]
#  plt.subplot(C,9,9*i+1)
#  plt.imshow(meanEdges[i], cmap='Reds')
#  for j in range(8):
#    plt.subplot(C,9,9*i+j+2)
#    plt.imshow(w[:,:,j], cmap='bone')
#    
#plt.show()


#Histogram of digit-weight dot products, adjust as needed:
#pts = df.flatPixelTestData()
#for wtClass in range(10):
#  for imgClass in range(10):
#    plt.subplot(10,10,1+wtClass*10+imgClass)
#    products = np.inner(pts[imgClass*1000:(imgClass+1)*1000],W[wtClass])
#    plt.hist(products, range=(-10,10))
#plt.show()