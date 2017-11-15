# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:27:25 2017

@author:    Bruce Goldfeder
            CSI 873
            Fall 2017
            HW #9 - KNN
"""

import os
import numpy as np
import operator

def ReadInFiles(path,trnORtst):
    # This reads in all the files from a directory filtering on what the file
    # starts with
    fullData = []
    fnames = os.listdir(path)
    for fname in fnames:
        if fname.startswith(trnORtst):
            print (fname)
            data = np.loadtxt(path + "\\" + fname)
            fullData.append(data)
    #numFiles = len (fullData)
    #print(numFiles)
   
    return fullData
    
def ReadInOneList(fullData,maxRows):
    # This function combines all of the data into one array for ease of use
    # It contains a capping ability to configure how many results to use
    allData = []
    numFiles = len (fullData)
    for j in range (numFiles):
        # allows for smaller data set sizes
        numRows = len (fullData[j])
        #print('numrows,maxrows ',numRows,maxRows)
        if (maxRows < numRows):
            numRows = maxRows
    
        for k in range(numRows):
            allData.append(fullData[j][k])
    return np.asarray(allData)

def MakeBinary(my_data):
    # This module converts the 0-255 to 0 or 1 binomial
    my_data[my_data > 0] = 1
    
    return my_data

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
    
    flag = False
    #wiList = weight(sqDistances,epsilon,flag)
    #weightedDist = np.multiply(wiList,distances)
    
    # I am multiplying after finding the distances
    sortedDists = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = answerData[sortedDists[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + weight(distances[sortedDists[i]],flag)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def weight(k2weight,flag):
    epsilon = 1.0
    # returns the weight array based on flag true means weight = 1
    # false = calculate the weight
    if flag:
        weightAmt = 1
    else:
        denom = k2weight**2 + epsilon
        weightAmt = 1 / denom
    
    return weightAmt

def driver(dpath,trnNum,tstNum):
    
    # Read in the Training data first
    datasetTrn = ReadInFiles(dpath,'train')
    my_data = ReadInOneList(datasetTrn,trnNum)
    
    # Convert the 0-255 to 0 through 1 values in data
    #my_data[:,1:] /= 255.0
    
    
    # randomize the rows for better training
    #np.random.shuffle(my_data)
    inNum,cols = my_data.shape    
    just_trn_data = my_data[:,1:]
    just_trn_data_bin = MakeBinary(just_trn_data)
    answerTrn = my_data[:,0]
    
    # Read in the test data
    #dpath2 = os.getcwd()+'\data3'
    dataset2 = ReadInFiles(dpath,'test')
    my_test = ReadInOneList(dataset2,tstNum) 
    
    tstNum,cols = my_test.shape
    #print('num rows ',tstNum)
    
    # Convert the 0-255 to 0 through 1 values in data
    my_test[:,1:] /= 255.0
    
    just_test_data = my_test[:,1:]
    just_test_data_bin = MakeBinary(just_test_data)
    answerTest = my_test[:,0]    
    
    # Run the KNN algorithm
    tstAccList = []
    k=7
    for i in range(tstNum):
        result = predictKNN(just_test_data_bin[i],just_trn_data_bin,answerTrn,k)
        #print('KNN is ',result,' answer is ',answerTest[i])
        if (result - answerTest[i] == 0):
            tstAccList.append(1)
        else:
            tstAccList.append(0)
    
    # Output the Test set accuracy
    right = sum(tstAccList)
    total = len(tstAccList)
    testAccuracy = right/total
    print('Final Test results of ',right,' out of ',total,' accuracy is ',testAccuracy)

if __name__ == "__main__":
    
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filepath", help="Folder path for data")
    parser.add_option("-t", "--train", dest="trnNum", help="Number of Training Images per Number")
    parser.add_option("-x", "--test", dest="tstNum", help="Number of Test Images per Number")

    options, args = parser.parse_args()
    
    if not options.filepath :
        print("Used default of data" )
        filepath = os.getcwd()+'\data3'
    else: filepath = options.filepath
    
    if not options.trnNum :
        print("Used default trnNum = 50" )
        trnNum = 50
    else: trnNum = int(options.trnNum)
    
    if not options.tstNum :
        print("Used default tstNum = 50" )
        tstNum = 50
    else: tstNum = int(options.tstNum) 
    
    # Call the driver function for KNN
    driver(filepath,trnNum,tstNum)