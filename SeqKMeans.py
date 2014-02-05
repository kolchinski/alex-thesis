import numpy as np
import matplotlib.pyplot as plt


class SeqKMeans:
  #trainPts should be a 2D numpy array of first-axis length (# of training points)
  def __init__(self, trainPts):
    self.trainPts = trainPts
    #self.trainMeans()
    #self.trainMeansGrowing()

  #Make new clusters when none is close enough
  #This doesn't work very well - better to focus on mixed number of clusters
  def trainMeansGrowing(self, clusterCutoff = 10.0):
    Ts = self.trainPts
    np.random.shuffle(Ts)
    N = len(Ts)
    self.sideLength = np.sqrt(len(self.trainPts[0]))
    s = self.sideLength
    self.means = [np.copy(Ts[0])]
    self.weights = [1]
    NEW_CLUSTER_CUTOFF = clusterCutoff
 
    for n in range(N):
      dists = self.dists_to_means(Ts[n])
      if np.min(dists) < NEW_CLUSTER_CUTOFF:
        closestCluster = np.argmin(dists)
        self.weights[closestCluster] += 1
        delta = (1.0 / self.weights[closestCluster]) * (Ts[n] - self.means[closestCluster])
        self.means[closestCluster] += delta  
      else:
        self.means.append(np.copy(Ts[n]))
        self.weights.append(1)
    
    print "Points per cluster: ", self.weights
    plt.close('all')
    numClusters = len(self.means)
    fig, axes = plt.subplots(1, numClusters, figsize=(0.7*numClusters,1)) 
    for i in range(numClusters):
      curMean = np.copy(self.means[i])
      curAxes = axes[i]
      curAxes.xaxis.set_ticks([])
      curAxes.yaxis.set_ticks([])
      curAxes.imshow(curMean.reshape(s,s), cmap='bone')
    plt.show()


  def trainMeans(self, numClusters):
    Ts = self.trainPts
    np.random.shuffle(Ts)
    N = len(Ts)
    K = numClusters
    self.sideLength = np.sqrt(len(self.trainPts[0]))
    s = self.sideLength

    #initialize means to first K points
    self.means = (Ts[:K]).astype(np.float64)

    #randomly initialize means
    #self.means = (np.random.rand(K,s*s) > 0.5) * 1.0
    
    #weights is number of examples previously factored into a cluster mean;
    #this influences how much subsequent examples assigned to that cluster
    #move its mean by
    self.weights = np.zeros(K)

    numGraphSteps = 10
    graphEvery = N / numGraphSteps

    plt.close('all')
    fig, axes = plt.subplots(numGraphSteps, K, figsize=(10,10)) 

    np.random.shuffle(Ts)
    for n in range(N):
      closestCluster = self.classify(Ts[n])
      self.weights[closestCluster] += 1
      
      delta = (1.0 / self.weights[closestCluster]) * (Ts[n] - self.means[closestCluster])
      #delta = (Ts[n] - self.means[closestCluster])

      self.means[closestCluster] += delta  
      if n % graphEvery == 0:
        for i in range(K):
          curMean = np.copy(self.means[i])
          curAxes = axes[n/graphEvery, i]
          curAxes.xaxis.set_ticks([])
          curAxes.yaxis.set_ticks([])
          curAxes.imshow(curMean.reshape(s,s), cmap='bone')

    print "Points per cluster: ", self.weights
    plt.show()


  #Use Euclidian distance instead
  def classify(self, p):
    #print self.dists_to_means(p)
    return np.argmin(self.dists_to_means(p))

  def dists_to_means(self,p):
    return np.apply_along_axis(np.linalg.norm,1,self.means-p)

  def showClusters(self):
    sideLength = np.sqrt(len(self.trainPts[0]))
    clusterGrids = self.means.reshape(self.numClusters,sideLength,sideLength)
    plt.close('all')
    for i in range(self.numClusters):
      plt.subplot(1,10,i+1)
      plt.gca().axes.get_xaxis().set_ticks([])
      plt.gca().axes.get_yaxis().set_ticks([])
      plt.imshow(clusterGrids[i], cmap='bone')
    plt.show()

  #If given N test points, M classes, expect sequentially N/M test points per class
  #However, for this to work, we have to figure out which cluster is which class
  #Nevertheless, confusion matrix is useful
  def test(self, testPts):
    N = len(testPts)
    K = self.numClusters
    errors = np.zeros(K)
    classifications = np.zeros((K,K))
    for n in range(N):
      trueClass = K * n / N
      classedAs = self.classify(testPts[n])
      if classedAs != trueClass: errors[trueClass] += 1
      classifications[trueClass,classedAs] += 1
    errorRates = 1.0 * errors / (N / K)
    #for i in range(K):
    #  print "Error rate {} for class {}".format(errorRates[i], i)
    #print "Overall error rate: {}".format(errorRates.mean())
    print classifications

#import digit_features as df
#skm1 = SeqKMeans(df.flatPixelTrainData(),10)
#skm1.trainMeans()
