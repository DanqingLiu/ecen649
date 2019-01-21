import csv
import matplotlib.pyplot as plt
import numpy as np

def loadCsv(filename):
   lines = csv.reader(open(filename, 'rt', encoding = 'utf-8'))
   next(lines) # to skip the first header line in csv file
   dataset = list(lines)
   for i in range(len(dataset)):
      dataset[i] = [float(x) for x in dataset[i]]
   return dataset

def storeDataforplot(dataset):
   xylabel1 = {}
   xylabel_1 = {}
   for i in range(3):
      xylabel1[i] = []
      xylabel_1[i] = []
   for i in range(len(dataset)):
      if (dataset[i][-1] == 1):
         xylabel1[0].append(dataset[i][0])
         xylabel1[1].append(dataset[i][1])
      elif (dataset[i][-1] == -1):
         xylabel_1[0].append(dataset[i][0])
         xylabel_1[1].append(dataset[i][1])
   return [xylabel1, xylabel_1]

# Make a prediction with weights
def predict(row, weights):
   activation = weights[0]
   for i in range(len(row)-1):
      activation += weights[i + 1] * row[i]
   return 1.0 if activation >= 0.0 else -1

# Estimate Perceptron weights using stochastic gradient descent
def trainWeights(train, lrate, nepoch):
   weights = [0.0 for i in range(len(train[0]))]
   for epoch in range(nepoch):
      sum_error = 0.0
      for row in train:
         prediction = predict(row, weights)
         error = row[-1] - prediction
         sum_error += error**2
         weights[0] = weights[0] + lrate * error
         for i in range(len(row)-1):
            weights[i + 1] = weights[i + 1] + lrate * error * row[i]
   return weights

def perceptron(train, test, lrate, nepoch):
   predictions = []
   weights = trainWeights(train, lrate, nepoch)
   for row in test:
      prediction = predict(row, weights)
      predictions.append(prediction)
   return(predictions)

def main():
   lrate = 0.1
   nepoch = 5
   trainfile = 'C:/3-TAMU/1----Study/3----Course/EE649-PatternRecognition/iris_train.csv'
   trainset = loadCsv(trainfile)
   [xylabel1, xylabel2] = storeDataforplot(trainset)

   # plt.plot(xylabel1[0], xylabel1[1], 'ro')
   # plt.plot(xylabel2[0], xylabel2[1], 'bx')
   x = np.linspace(0, 10, 1000)
   weights = trainWeights(trainset, lrate, nepoch)
   plt.plot(x, -(weights[1] * x + weights[0])/weights[2], linestyle='solid')
   plt.axis([0, 8, 0, 3])


   testfile = 'C:/3-TAMU/1----Study/3----Course/EE649-PatternRecognition/iris_test.csv'
   testset = loadCsv(testfile)
   predictions = perceptron(trainset, testset, lrate, nepoch)
   [testlabel1, testlabel2] = storeDataforplot(testset)
   plt.plot(testlabel1[0], testlabel1[1], 'ro')
   plt.plot(testlabel2[0], testlabel2[1], 'bx')
   plt.show()

main()

