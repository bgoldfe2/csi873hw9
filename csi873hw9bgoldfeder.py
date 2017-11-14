# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:27:25 2017

@author:    Bruce Goldfeder
            CSI 873
            Fall 2017
            HW #9 - KNN
"""

import numpy as np
import operator


# Takes the x_q input to be tested, the training data array, the answer array,
# and the value for k
def predictKNN(testData, trainData, answerData, k):
    # First find the Euclidean Distance 
    dataSetSize = trainData.shape[0]
    diffMat = np.tile(testData, (dataSetSize,1)) - trainData
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    # Then find the weight
    # first time through the weight is 1
    # epsilon is always 1
    epsilon = 1
    wiList = weight(sqDistances,epsilon)
    weightedDist = np.multiply(wiList,distances)
    
    # I am multiplying after finding the distances
    sortedDists = weightedDist.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = answerData[sortedDists[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def weight(sqDistances,epsilon,flag):

    # returns the weight array based on flag true means weight = 1
    # false = calculate the weight
    if flag:
        numDist = sqDistances.shape
        weightList = np.ones(numDist)
    else:
        weightList = np.divide(1,np.add(sqDistances,epsilon))
    
    return weightList
