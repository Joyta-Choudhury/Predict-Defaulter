from __future__ import print_function

import math
import sys
from operator import add
from pyspark import SparkContext

#For Naive Bayes
#dictionary containing number of rows for each target value
count_c = dict()
prob_c = dict()

#total rows in data
totalRows = 0


#Delimeter to separate column value from targetValue
delim = '_#_'
tdict = dict()

#Global list for training data
train = []
k = sys.argv[1]

#Target Column heading name (Same name should be used/indexed in catCols dictionary)
pred_var = 'loan_status'

catCols = dict()
catCols['loan_amnt'] = 7
catCols['int_rate'] = 8
catCols['annual_inc'] = 9 
catCols['dti'] = 10
catCols['open_acc'] = 11
catCols['revol_bal'] = 12
catCols['total_acc'] = 13
catCols['revol_util'] = 14
catCols['delinq_amnt'] = 15
catCols['loan_status'] = 5

def processTrain(row):
    tokens = str(row).split(',') 
    dat = dict()
    for col in catCols:
        if col != pred_var:
            dat[col] = float(tokens[catCols[col]])
        else:
            dat[col] = tokens[catCols[col]]
    return dat

def predict(row):
    tokens = str(row[0]).split(',') 
    distances = []
    
    for trRow in train:
        distance=0
        for key in trRow:
            if key != pred_var:
                distance += (trRow[key] - float(tokens[catCols[key]]))**2
        distances.append((math.sqrt(distance),tokens[catCols[pred_var]]))
    
    final_tuples = sorted(distances, key=lambda x: x[0])[0:k]
    
    outDict = dict()
    for tup in final_tuples:
        if tup[1] not in outDict:
            outDict[tup[1]] = 0
        
        outDict[tup[1]] += 1
    
    gkey = -1
    gval = 0
    for item in outDict:
        if outDict[item] > gval:
            gkey = item
            gval = outDict[item]
    
    return gkey


if __name__ == "__main__":
    #trainFile = 'D:\\Study\\1 DSBA\\Sem II\\Cloud Computing\\project\\work\\data\\trdata.csv'
    #testFile = 'D:\\Study\\1 DSBA\\Sem II\\Cloud Computing\\project\\work\\data\\tsdata.csv'
    trainFile = sys.argv[2]
    testFile = sys.argv[3]
    
    sc = SparkContext(appName="KNN")
    #spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

    #Collect frequency of each occurance of categorical values including target and store in tuples
    traincols = sc.textFile(trainFile).map(lambda x: processTrain(x))
    train = traincols.collect()
    print('Training Complete.')
    
    testLst = sc.textFile(testFile).map(lambda x: predict([x]))   
    output = testLst.collect()
    print('Predictions Complete')
    
    import pandas as pd
    tst = pd.read_csv(testLst,header=None)

    print ('KNN Model Accuracy: ',end='')
    print((tst[5] == output).sum()/tst.shape[0])
    sc.stop()