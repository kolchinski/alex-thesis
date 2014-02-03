from sklearn import svm
import digit_features as df
import numpy as np

# Quadratic time since pairwise SVMs - implemented with libSVM
def oneVsOne():
  X = df.flatEdgeTrainData()
  Y = np.arange(1000) / 100
  clf = svm.SVC()
  clf.fit(X,Y)
  predictions = clf.predict(df.flatEdgeTestData())
  testTruth = np.arange(10000) / 1000
  print "Percent error: ", np.sum(testTruth != predictions) / 10000.0
  classifications = np.zeros((10,10))
  for i in range(10000):
      classifications[i/1000][predictions[i]] += 1
  print(classifications)

# Implemented with liblinear
def oneVsRest():
  X = df.flatEdgeTrainData()
  Y = np.arange(1000) / 100
  clf = svm.LinearSVC()
  clf.fit(X,Y)
  predictions = clf.predict(df.flatEdgeTestData())
  testTruth = np.arange(10000) / 1000
  print "Percent error: ", np.sum(testTruth != predictions) / 10000.0
  classifications = np.zeros((10,10))
  for i in range(10000):
      classifications[i/1000][predictions[i]] += 1
  print(classifications)


