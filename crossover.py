import random
import copy
import math

# 3.3.2 Size Matters Crossover in report
def cxProbSwitch(ind1, ind2):
    """ Switch positions of the squares of between the
    individuals.
    Smallest always, least smallest with high prob. etc. 
    """
    nInd1 = copy.copy(ind1)
    nInd2 = copy.copy(ind2)

    numSquares = len(ind1) #/ 2

    flipProbs = (random.random() < 0.5)
    
    for i in xrange(0, numSquares):
        if random.random() < flipProbs - (float(i) / numSquares):
            index = i

            nInd1[index] = ind2[index]
            
            nInd2[index] = ind1[index]
    
    return nInd1, nInd2

# 3.3.2 Weighted Average Crossover in report
def cxWeightedAverage(ind1, ind2):
    nInd1 = copy.copy(ind1)
    nInd2 = copy.copy(ind2)
    
    for i in xrange(0, len(nInd1)):
        if random.random() < 0.5:
            nInd1[i] = int(round((2*ind1[i] + ind2[i]) / 3.0))
        if random.random() < 0.5:
            nInd2[i] = int(round((ind1[i] + 2*ind2[i]) / 3.0))

    return nInd1, nInd2

# 3.3.2 Quantative Crossover in report
def cxNumOverlap(ind1, ind2, square_size):
    """ Pick least overlap squares in nInd1 and
    most overlap squares in nInd2.
    """
    nInd1 = copy.copy(ind1)
    nInd2 = copy.copy(ind2)
    
    for i in range(0,len(ind1)-1,2):
        x1_i = ind1[i]
        y1_i = ind1[i+1]
        x2_i = ind2[i]
        y2_i = ind2[i+1]

        overlap1_sum = 0
        overlap2_sum = 0
        for j in range(0,len(ind1)-1,2):
            if i == j:
                continue
            x1_j = ind1[j]
            y1_j = ind1[j+1]
            x2_j = ind2[j]
            y2_j = ind2[j+1]
            
            x1_overlap = max(0, min(x1_i + square_size(i), x1_j + square_size(j)) - max(x1_i, x1_j))
            y1_overlap = max(0, min(y1_i + square_size(i), y1_j + square_size(j)) - max(y1_i, y1_j))
            x2_overlap = max(0, min(x2_i + square_size(i), x2_j + square_size(j)) - max(x2_i, x2_j))
            y2_overlap = max(0, min(y2_i + square_size(i), y2_j + square_size(j)) - max(y2_i, y2_j))

            overlap1_area = x1_overlap * y1_overlap
            overlap2_area = x2_overlap * y2_overlap
            overlap1_sum += (overlap1_area > 0)
            overlap2_sum += (overlap2_area > 0)

        if overlap1_sum < overlap2_sum:
            nInd1[i] = ind1[i]
            nInd1[i+1] = ind1[i+1]
            nInd2[i] = ind2[i]
            nInd2[i+1] = ind2[i+1]
        else:
            nInd1[i] = ind2[i]
            nInd1[i+1] = ind2[i+1]
            nInd2[i] = ind1[i]
            nInd2[i+1] = ind1[i+1]
    return nInd1, nInd2

