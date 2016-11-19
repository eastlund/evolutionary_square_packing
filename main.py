# -*- coding: utf-8 -*-
# -- This file runs the experiments over a specific range of n configurations --

import sys
import matplotlib.pyplot as plt
import time
import numpy as np

import os
import errno

from ga import GA
from pso import PSO
import experiments

REPS=50
DEBUG=True
n_range=[3,9]

# List of provable smallest working s for different n values. Can be used to terminate search when optimality has been reached.
fraud = [0, 0, 3 , 5 , 7 , 9 , 11 , 13 , 15 , 18 , 21 , 24 , 27 , 30 , 33 , 36 , 39 , 43 , 47 , 50 , 54 , 58 , 62 , 66] 

def timer(fun):
    startTime = time.time()
    retVal = fun()
    elapsedTime = time.time() - startTime
    return retVal, elapsedTime

# creates path if not already created
def path_creation(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main(argv):
    path_creation("results/pso");
    path_creation("results/ga");

    for gaTest in experiments.gaTests:
        name = gaTest["name"]
        conf = gaTest["conf"]
        if DEBUG:
            print name
            
        for n in xrange(n_range[0], n_range[1]):
            times = []
            bestSArr = []
            nGens = []
            output = {}
            solutionsFound = 0
            for i in range(0,REPS):
                conf["n"] = n
                if DEBUG:
                    print gaTest
                g = GA(**conf)
            
                res, time = timer(lambda : g.run())
                
                bestS = res["bestS"]
                bestInd = res["bestInd"]
                backupBest = res["backupBest"]
                nGen = res["nGens"]
                fitness = bestInd.fitness.values
                
                bestSArr.append(bestS)
                times.append(time)
                nGens.append(nGen)

                if fitness[0] == 0.0 and bestS == fraud[n]:
                    solutionsFound += 1

                if DEBUG:
                    sys.stdout.write('\t')
                    print bestS, bestInd, backupBest, nGens, fitness, solutionsFound

            meanTimes = np.mean(times)
            stdTimes = np.std(times)
            meanGens = np.mean(nGens)
            stdGens = np.std(nGens)
            meanS = np.mean(bestSArr)
            stdS = np.std(bestSArr)
            output["meanTimes"] = meanTimes
            output["stdTimes"] = stdTimes
            output["meanGens"] = meanGens
            output["stdGens"] = stdGens
            output["meanS"] = meanS
            output["stdS"] = stdS
            output["solutionPercentage"] = float(solutionsFound) / REPS

            output["n"] = n
            with open('results/ga/ga' + "_" + str(name) + "_n=" + str(n) + "_reps=" + str(REPS) + ".data", 'w+') as the_file: 
                    the_file.write(str(output))
    for psoTest in experiments.psoTests:
        name = psoTest["name"]
        conf = psoTest["conf"]
        if DEBUG:
            print name
        for n in xrange(n_range[0], n_range[1]):
            times = []
            bestSArr = []
            nGens = []
            output = {}
            solutionsFound = 0
            for i in range(0,REPS):
                conf["n"] = n
                if DEBUG:
                    print psoTest
                    print conf
                pso = PSO(**conf)
                res, time = timer(lambda : pso.run())
                bestS = res["bestS"]
                bestInd = res["bestInd"] 
                nGen = res["nGens"]
                fitness = res["bestFitness"]
                
                bestSArr.append(bestS)
                times.append(time)
                nGens.append(nGen)

                if fitness[0] == 0.0 and bestS == fraud[n]:
                    solutionsFound += 1
                
                if DEBUG:
                    sys.stdout.write('\t')
                    print bestS, bestInd, fitness[0], nGens, solutionsFound
                
            meanTimes = np.mean(times)
            stdTimes = np.std(times)
            meanGens = np.mean(nGens)
            stdGens = np.std(nGens)
            meanS = np.mean(bestSArr)
            stdS = np.std(bestSArr)
            output["meanTimes"] = meanTimes
            output["stdTimes"] = stdTimes
            output["meanGens"] = meanGens
            output["stdGens"] = stdGens
            output["meanS"] = meanS
            output["stdS"] = stdS
            output["solutionPercentage"] = float(solutionsFound) / REPS

            output["n"] = n
            
            with open('results/pso/pso' + "_" + str(name) + "_n=" + str(n) + ".data", 'w+') as the_file: 
                    the_file.write(str(output))
            
if __name__ == "__main__":
    main(sys.argv[1:])
