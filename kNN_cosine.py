#########################################  
# kNN: k Nearest Neighbors  
  
# Input:      newInput: vector to compare to existing dataset (1xN)  
#             dataSet:  size m data set of known vectors (NxM)  
#             labels:   data set labels (1xM vector)  
#             k:        number of neighbors to use for comparison   
              
# Output:     the most popular class label  
#########################################  
  
from numpy import *  
import operator
import math
import numpy as np
from common_tools import cosine_distance_numpy


# classify using kNN  
def kNNClassify(newInput, dataSet, labels, k): 
    global distance
    distance = [0]* dataSet.shape[0]
    for i in range(dataSet.shape[0]):
        distance[i] = cosine_distance_numpy(newInput, dataSet[i])
  
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)  
  
    classCount = {}     # define a dictionary (can be append element)  
    for i in range(k):
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]  
  
        ## step 4: count the times labels occur  
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
  
    ## step 5: the max voted class will return  
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex
    #return sortedDistIndices   
