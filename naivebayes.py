import csv
import math
import collections
import numpy

# 1 Use the train set to estimate the probabilities for getting each digit
# P(Y=0), P(Y=1), P(Y=2), P(Y=3), ..., P(Y=9)
#  and the conditional probabilities of individual pixels given the values of the digit.
# P(X1=0|Y=0), P(X1=1|Y=0), P(X1=2|Y=0), ..., P(X1=255|Y=0),
# P(X1=0|Y=1), P(X1=1|Y=1), P(X1=2|Y=1), ..., P(X1=255|Y=1),
# ...
# P(X1=0|Y=9), P(X1=1|Y=9), P(X1=2|Y=9), ..., P(X1=255|Y=9),
# P(X2=0|Y=0), P(X2=1|Y=0), P(X2=2|Y=0), ..., P(X2=255|Y=0),
# ...,
# P(X784=0|Y=0), P(X784=1|Y=0), P(X784=2|Y=0), ..., P(X784=255|Y=0),
# ...
# Record the results into a table.

# Load data from file.
def loadCsv(filename):
   lines = csv.reader(open(filename, 'rt', encoding = 'utf-8'))
   dataset = list(lines)
   for i in range(len(dataset)):
      dataset[i] = [int(x) for x in dataset[i]]
   return dataset

# Store data to a dictionary sorted by key Y from 0 to 9
def storeData(dataset):
   separated = {}
   for i in range(len(dataset)):
      vector = dataset[i]
      if (vector[0] not in separated):
         separated[vector[0]] = []
      separated[vector[0]].append(vector)
   sorteddict = collections.OrderedDict(sorted(separated.items()))
   return sorteddict

def countY(dataset):
   count = {}
   for i in range(len(dataset)):
      vector = dataset[i]
      if (vector[0] not in count):
         count[vector[0]] = 1
      count[vector[0]] = count[vector[0]] + 1
   return count

# estimate the probabilities for getting each digit
def calculateProbability(dataset):
   probability = {}
   digitscount = countY(dataset)
   total = len(dataset)
   for y in digitscount:
      probability[y] = digitscount[y]/total
   sortedy = collections.OrderedDict(sorted(probability.items()))
   return sortedy

# example  y=0: [x1=0, x2=3, x3=2, ..., x784=3] [x1=1, x2=255, x3=244, ..., x784=0], ...
# => x1=0: num, x1=1: num, ..., x1=255: num, x2=0: num, x2=1: num, ...
def countXi(fixedylist):
   countxi = [[0 for x in range(256)] for y in range(784)]
   for j in range(len(fixedylist)):
      vector = fixedylist[j]
      for i in range(784):
         countxi[i][vector[i]] = countxi[i][vector[i]] + 1
   return countxi

# the conditional probabilities of individual pixels given the values of the digit
# For a fixed Xi, there are 10 x 256 conditional probabilities
def calculateConditionalProbabilities(dataset):
   conditionalprobabilities = {}
   pixelsdata = storeData(dataset)
   county = countY(dataset)
   for key, value in pixelsdata.items():
      conditionalprobabilities[key] = (numpy.array(countXi(value)) + 1) / (county[key] + 256)
   return conditionalprobabilities

# Given a vector [x1, x2, x3, ..., x784], calculate {[0: p(x1,x2,...,x784|y=0)],[1: p(x1,x2,...,x784|y=1)],...,[9: p(x1,x2,...,x784|y=9)]}
def calculateClassProbabilities(summaries, inputvector):
   probabilities = {}
   testvector = []
   for k in range(784):
      testvector.append(inputvector[k+1])
   for y in range(10):
      probabilities[y] = 1
      for i in range(len(testvector)):
         probabilities[y] = probabilities[y] * summaries[y][i][testvector[i]]
   return probabilities

def predict(summaries, inputvector):
   probabilities = calculateClassProbabilities(summaries, inputvector)
   bestlabel, bestprob = None, -1
   for classvalue, probability in probabilities.items():
      if bestlabel is None or probability > bestprob:
         bestprob = probability
         bestlabel = classvalue
   return bestlabel

#4 Make Predictions
def getPredictions(summaries, testset):
   predictions = []
   for i in range(len(testset)):
      result = predict(summaries, testset[i])
      predictions.append(result)
   return predictions

#5 Get Accuracy
def getAccuracy(testset, predictions):
   correct = 0
   for i in range(len(testset)):
      if testset[i][-1] == predictions[i]:
         correct += 1
   return (correct / float(len(testset))) * 100.0


def main():
   trainfile = 'C:/3-TAMU/1----Study/3----Course/EE649-PatternRecognition/mnist_train.csv'
   trainset = loadCsv(trainfile)
   #  a) your estimate of the probabilities of the ten hand-written digits based on the training set;
   p_y = calculateProbability(trainset)
   # print('Digits probability {0}'.format(p_y))

   #  b) conditional probabilities of the pixels given the hand-written digits and your estimates based on the training set
   summaries = calculateConditionalProbabilities(trainset)
   # print('Conditional probabilities {0}'.format(summaries))

   testfile = 'C:/3-TAMU/1----Study/3----Course/EE649-PatternRecognition/mnist_test.csv'
   testset = loadCsv(testfile)
   # test model
   predictions = getPredictions(summaries, testset)
   accuracy = getAccuracy(testset, predictions)
   print('Accuracy: {0}%'.format(accuracy))


main()

