from operator import pos
import numpy as np
from math import *

def rotateX( theta, coords):
    matrix = np.array([[1, 0, 0], [0, np.cos(theta), -1 * np.cos(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(matrix, coords)

def euclideanNorm( vector):
    total = 0
    for val in vector:
      total = total + pow(val, 2)
    return sqrt(total)

def angleBetweenVectors( v1, v2):
    return acos(np.dot(v1,v2) / (euclideanNorm(v1) * euclideanNorm(v2)))

def projectionOntoPlane( vector, planeNormal):
    return vector - projection(vector, planeNormal)

def projection( v1, v2): #Projects v1 onto v2
    return np.multiply((np.dot(v1,v2) / pow(euclideanNorm(v2),2)), v2)

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def smoothed_predicted_q(predictionHistory):

    positives = np.zeros((3))
    negatives = np.zeros((3))

    sumPositives = np.zeros((3))
    sumNegatives = np.zeros((3))

    for q in predictionHistory:
        positives += q >= 0
        sumPositives += q * (q >= 0)
        
        negatives += q < 0
        sumNegatives += q * (q<0)
    
    result = np.zeros((3))

    for i in range(len(result)):
        if positives[i] > negatives[i]:
            result[i] = sumPositives[i]/positives[i]
        elif positives[i] < negatives[i]:
            result[i] = sumNegatives[i]/negatives[i]
        else:
            result[i] = (sumPositives[i] + sumNegatives[i])/(positives[i] + negatives[i])
    return result
            