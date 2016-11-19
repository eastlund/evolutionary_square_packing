# -- Setup experiments in this file to be used in main.py --

import random

from deap import base
from deap import creator
from deap import tools

import sys
import math
import time
import copy
import crossover
import mutation
import ga # TODO: bad practice! (move fitness functions)
import pso

import numpy as np

def tourn5(toolbox, self):
    return toolbox.register("select", tools.selTournament, tournsize=5) # tournsize=1 => random selection

def tourn10(toolbox, self):
    return toolbox.register("select", tools.selTournament, tournsize=10)

def tourn25(toolbox, self):
    return toolbox.register("select", tools.selTournament, tournsize=25)

def tourn50(toolbox, self):
    return toolbox.register("select", tools.selTournament, tournsize=50)

def twoPointCrossover(toolbox, self):
    return toolbox.register("mate", tools.cxTwoPoint)

gaTests = [{"name" : "areaOverlap_mutpb="+str(mutFactor)+"_cxpb="+str(cxFactor),
               "conf" : {"fitnessFun" : ga.areaOverlap,
                         "fraudFlag" : True,
                         "ignoreSmallest" : True,
                         "forcePlacement" : True, 
                         "mutpb" : mutFactor,
                         "cxpb" : cxFactor}} for mutFactor in np.arange(0.1, 1.1, 0.1) for cxFactor in np.arange(0.0, 1.1, 0.1) if not(mutFactor == 0 and cxFactor == 0)]

## Do not fix n
psoInitialTests = [{"name" : "areaOverlap_cogn="+str(cognFactor)+"_social="+str(socialFactor),
             "conf" : {"fitnessFun" : pso.areaOverlapFloat,
                       "fraudFlag" : True,
                       "ignoreSmallest" : True,
                       "cognitiveFactor" : cognFactor,
                       "socialFactor" : socialFactor}} for cognFactor in np.arange(1.0, 2.05, 0.1) for socialFactor in np.arange(1.0, 2.05, 0.1)] + [{"name" : "costlyCost_cogn="+str(cognFactor)+"_social="+str(socialFactor),
             "conf" : {"fitnessFun" : pso.costlyCostFloat,
                       "fraudFlag" : True,
                       "ignoreSmallest" : True,
                       "cognitiveFactor" : cognFactor,
                       "socialFactor" : socialFactor}} for cognFactor in np.arange(1.0, 2.05, 0.1) for socialFactor in np.arange(1.0, 2.05, 0.1)]

psoGenPop = [{"name" : "costlier_cogn="+str(1.9)+"_social="+str(1.6)+"_gen="+str(gens)+"_pop="+str(pop),
             "conf" : {"fitnessFun" : pso.costlierCostFloat,
                       "fraudFlag" : True,
                       "ignoreSmallest" : True,
                       "cognitiveFactor" : 1.9,
                       "socialFactor" : 1.6,
                       "generations" : gens,
                       "population" : pop}} for (gens, pop) in [(300, 100), (500, 50), (2500, 20)]] + [{"name" : "costly_cogn="+str(1.9)+"_social="+str(1.8)+"_gen="+str(gens)+"_pop="+str(pop),
                                                                                                        "conf" : {"fitnessFun" : pso.costlyCostFloat,
                                                                                                                  "fraudFlag" : True,
                                                                                                                  "ignoreSmallest" : True,
                                                                                                                  "cognitiveFactor" : 1.9,
                                                                                                                  "socialFactor" : 1.8,
                                                                                                                  "generations" : gens,
                                                                                                                  "population" : pop}} for (gens, pop) in [(300, 100), (500, 50), (2500, 20)]] + [{"name" : "areaOverlap_cogn="+str(2.0)+"_social="+str(1.7)+"_gen="+str(gens)+"_pop="+str(pop),
                                                                                                                                                                                                   "conf" : {"fitnessFun" : pso.areaOverlapFloat,
                                                                                                                                                                                                             "fraudFlag" : True,
                                                                                                                                                                                                             "ignoreSmallest" : True,
                                                                                                                                                                                                             "cognitiveFactor" : 2.0,
                                                                                                                                                                                                             "socialFactor" : 1.7,
                                                                                                                                                                                                             "generations" : gens,
                                                                                                                                                                                                             "population" : pop}} for (gens, pop) in [(300, 100), (500, 50), (2500, 20)]]

psoTests = psoGenPop

