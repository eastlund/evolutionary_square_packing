# -*- coding: utf-8 -*-

import operator
import random

import numpy

from deap import base
from deap import creator
from deap import tools

import sys
import time
import math

# Force the placement of the biggest square or not by default
defaultForcePlacement = True
# Ignore the smallest square (1x1) or not by default
defaultIgnoreSmallest = True

fraud = [0, 0, 3 , 5 , 7 , 9 , 11 , 13 , 15 , 18 , 21 , 24 , 27 , 30 , 33 , 36 , 39 , 43 , 47 , 50 , 54 , 58 , 62 , 66] # list of provable smallest working s for different n values.

# We can't have a smaller enclosing square side than this
def minimal_s(n):
    return int(math.ceil(math.sqrt((n*(n+1)*(2*n+1)) / 6.0)))

def maximalS(n): ## Better than maximal_s
    if n < 1:
        return 0
    else:
        return n + maximalS_(n-1)

def maximalS_(n):
    if n < 1:
        return 0
    else:
        return n + maximalS_(n-3)

# 3.2.3 Grid fit in report            
def costlierCost(individual, tb):
    s = tb.s()
    square_Size = tb.squareSize
    
    grid = [[0 for _ in range(s)] for _ in range(s)]
    # Fill grid for all squares
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        
        for x_c in range(max(0,min(s,x_i)), min(s,x_i + square_Size(i))):
                for y_c in range(max(0,min(s,y_i)), min(s,y_i + square_Size(i))):
                        grid[x_c][y_c] = 1            

    overlap_sum = 0
    overlapping_squares = []
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        # positive overflow
        x_non_overflow = max(0, min(x_i + square_Size(i), 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + square_Size(i), 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (square_Size(i))**2 - non_overflow_area
  
        overlap_sum += 3*overflow_sum
        
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + square_Size(i), x_j + square_Size(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + square_Size(i), y_j + square_Size(j)) - max(y_i, y_j))
            overlap_area = x_overlap * y_overlap
            if overlap_area > 0:
                overlapping_squares.append(i)
            overlap_sum += 2*overlap_area
        
    
    # sum costs for getting out of overlap    
    for i in overlapping_squares:
        x_i = individual[i]
        y_i = individual[i+1]
        distances = [s]
        
        ## left direction
        pos = x_i
        y_middle = y_i + square_Size(i)/2
        steps = 0
        # Check if we are outside the grid
        if not (pos < 0 or pos >= s or y_middle < 0 or y_middle >= s):
            while (pos >= 0):
                if grid[pos][y_middle] == 0:
                    break
                pos -= 1
                steps += 1
            # make sure we stayed in the square
            if not (pos < 0):
                distances.append(steps)
        
        ## right direction
        pos = x_i + square_Size(i) - 1
        y_middle = y_i + square_Size(i)/2
        steps = 0
        if not (pos < 0 or pos >= s or y_middle < 0 or y_middle >= s):
            while (pos <= s-1):
                if grid[pos][y_middle] == 0:
                    break
                pos += 1
                steps += 1
            # make sure we stayed in the square
            if not (pos > s-1):
                distances.append(steps)
            
        ## down direction
        pos = y_i
        x_middle = x_i + square_Size(i)/2
        steps = 0
        if not (pos < 0 or pos >= s or x_middle < 0 or x_middle >= s):
            while (pos >= 0):
                if grid[x_middle][pos] == 0:
                    break
                pos -= 1
                steps += 1
            # make sure we stayed in the square       
            if not (pos < 0):
                distances.append(steps)
            
        # up direction
        pos = y_i + square_Size(i) - 1
        x_middle = x_i + square_Size(i)/2
        steps = 0
        if not (pos < 0 or pos >= s or x_middle < 0 or x_middle >= s):
            while (pos <= s-1):
                if grid[x_middle][pos] == 0:
                    break
                pos += 1
                steps += 1
            # make sure we stayed in the square
            if not (pos > s-1):
                distances.append(steps)
                
        overlap_sum += min(distances)
            
    return overlap_sum,

# PSO requires the positions to be floats during calculations
def costlierCostFloat(individual, tb):
    intividual = map(int,map(round, individual))
    return costlierCost(intividual, tb)
        
# 3.2.2 Distance Fit in report.
def costlyCost(individual, tb):
    s = tb.s()
    square_Size = tb.squareSize

    overflow_areas = 0
    overlap_areas = 0
    overlap_distances = 0
    # For all squares
    for i in range(0,len(individual)-1,2):
        x_i = individual[i] # x coordinate for the (i/2 + 1)th square
        y_i = individual[i+1] # y coordinate -||-
        
        # positive overflow -- alex kod
        x_non_overflow = max(0, min(x_i + square_Size(i), 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + square_Size(i), 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (square_Size(i))**2 - non_overflow_area
            
        overflow_areas += overflow_sum
        
        # for all squares which are larger than the current one, do they overlap?
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + square_Size(i), x_j + square_Size(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + square_Size(i), y_j + square_Size(j)) - max(y_i, y_j))
            # area of the overlap
            overlap_area = x_overlap * y_overlap
            # how far do we need to move the smaller square to not overlap?
            distances = [0.0]
            if overlap_area > 0:
                if x_i < x_j:
                    distances.append(x_i+square_Size(i)-x_j)
                elif x_j + square_Size(j) < (x_i + square_Size(i)):
                    distances.append(x_j+square_Size(j)-x_i)
                else:
                    distances.append(x_i - x_j + square_Size(i))
                    distances.append(x_j + square_Size(j) - (x_i + square_Size(i)) + square_Size(i))
                if y_i < y_j:
                    distances.append(y_i+square_Size(i)-y_j)
                elif (y_j + square_Size(j)) < (y_i + square_Size(i)):
                    distances.append(y_j+square_Size(j)-y_i)
                else:
                    distances.append(y_i - y_j + square_Size(i))
                    distances.append(y_j + square_Size(j) - (y_i + square_Size(i)) + square_Size(i))
            overlap_areas += overlap_area
            overlap_distances += min(distances)
    
    return (3*overflow_areas + 2*overlap_areas + overlap_distances),

# PSO requires the positions to be floats during calculations
def costlyCostFloat(individual, tb):
    intividual = map(round, individual)
    return costlyCost(intividual, tb)

# 3.2.1 Enumerative Overlap in report.
def numberOverlap(individual, tb):
    s = tb.s()
    square_Size = tb.squareSize

    overlap_sum = 0
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        x_non_overflow = max(0, min(x_i + (i/2)+1, 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + (i/2)+1, 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = ((i/2)+1)**2 - non_overflow_area
        overlap_sum += (overflow_sum > 0)
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + (i/2)+1, x_j + (j/2)+1) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + (i/2)+1, y_j + (j/2)+1) - max(y_i, y_j))
            overlapArea = x_overlap * y_overlap
            overlap_sum += (overlapArea > 0)
    return overlap_sum,

# PSO requires the positions to be floats during calculations
def numberOverlapFloat(individual, tb):
    intividual = map(round, individual)
    return numberOverlap(intividual, tb)

# The square size of square i
def square_Size(i, ignoreSmallest=True):
    return (i/2) + 1 + ignoreSmallest ## if not skipping 1x1 square.

# Gives the side length of the enclosing square
def sSize(individual, toolbox=None):
    ignoreSmallest = toolbox.ignoreSmallest()
    squareSize = toolbox.squareSize
    min_h = 0; min_w = 0 ## fix to 0?
    max_h = 0
    max_w = 0
    for i in range(0, len(individual)-1,2):
        if individual[i] + squareSize(i) > max_w:
            max_w = individual[i] + squareSize(i)
        if individual[i+1] + squareSize(i) > max_h:
            max_h = individual[i+1] + squareSize(i)

    h = max_h - min_h
    w = max_w - min_w
    return max(minimal_s(len(individual)/2 + ignoreSmallest), max(h,w))

# 3.2.1 Area Overlap in report.
def areaOverlap(individual, tb):
    s = tb.s()
    square_Size = tb.squareSize
    
    overlap_sum = 0
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        # positive overflow
        x_non_overflow = max(0, min(x_i + square_Size(i), 0 + tb.s()) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + square_Size(i), 0 + tb.s()) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (square_Size(i))**2 - non_overflow_area
                        
        overlap_sum += 2*overflow_sum
        
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + square_Size(i), x_j + square_Size(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + square_Size(i), y_j + square_Size(j)) - max(y_i, y_j))
            overlapArea = x_overlap * y_overlap
            overlap_sum += overlapArea
    return overlap_sum,

def areaOverlapFloat(individual, tb):
    intividual = map(round, individual)
    return areaOverlap(intividual, tb)


#----------

# Initializes particles with domain ranges, positions and velocities
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

# Particle update rule (gbest)
def updateParticle(part, best, indW, socW):
    u1 = (random.uniform(0, indW) for _ in range(len(part)))
    u2 = (random.uniform(0, socW) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))
    
    
class PSO:
    def defaultAttrCoord(toolbox, self):
        if self.forcePlacement:
            return toolbox.register("attrCoord", random.randint, self.n, self.s-2) # TODO: make custom if possible
        else:
            return toolbox.register("attrCoord", random.randint, 0, self.s-2)

    def defaultStructureInit(toolbox, self):
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.attrCoord, (self.n-self.ignoreSmallest)*2)

    def __init__(self,
                 dynamic=False,
                 n=7,
                 s=0,
                 debug=False,
                 ignoreSmallest=defaultIgnoreSmallest,
                 forcePlacement=defaultForcePlacement,
                 generations=500,
                 population=50,
                 fitnessFun=costlierCostFloat,
                 fraudFlag=False,
                 cognitiveFactor=1.49,
                 socialFactor=1.49):
        self.dynamic = dynamic
        self.n = n
        if s == 0:
            self.s = fraud[self.n] if fraudFlag else maximalS(self.n)
        else:
            self.s = s
        self.debug = debug
        self.generations = generations
        self.population = population
        
        self.fitnessFun = fitnessFun

        self.cognitiveFactor = cognitiveFactor
        self.socialFactor = socialFactor
        
        self.ignoreSmallest = ignoreSmallest
        self.forcePlacement = forcePlacement

        self.fraud = fraudFlag

    def run(self):
        if self.debug:
            print ("n:", self.n)
            print ("s:", self.s)

        if self.dynamic:
            self.s = fraud[self.n]

        best_s = self.s
        bestfit = 0
        best_ind = []
        stuck_counter = 0
        is_stuck = False
        shakes = 0
        backup_best = [] # Used with escaper, backups old best
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Particle",
                       list,
                       fitness=creator.FitnessMin,
                       speed=list,
                       smin=None,
                       smax=None,
                       best=None)

        toolbox = base.Toolbox()
        
        def getS():
            return self.s

        def getN():
            return self.n

        def getIgnoreSmallest():
            return self.ignoreSmallest

        def getForcePlacement():
            return self.forcePlacement
        
        # Register fields and function in toolbox here. #
        toolbox.register("squareSize", square_Size, ignoreSmallest=self.ignoreSmallest)
        toolbox.register("sSize", sSize, toolbox=toolbox)
        toolbox.register("s", getS)
        toolbox.register("n", getN)

        toolbox.register("ignoreSmallest", getIgnoreSmallest)
        toolbox.register("forcePlacement", getForcePlacement)
        
        toolbox.register("particle",
                         generate,
                         size=(self.n-self.ignoreSmallest)*2,
                         pmin=0,
                         pmax=self.s-1,
                         smin=-2,
                         smax=2)
        
        toolbox.register("population",
                         tools.initRepeat,
                         list,
                         toolbox.particle)

        toolbox.register("update",
                         updateParticle,
                         indW=self.cognitiveFactor,
                         socW=self.socialFactor)
        
        toolbox.register("evaluate",
                         self.fitnessFun,
                         tb=toolbox)
        
        pop = toolbox.population(n=self.population)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields
        
        total_best = None
        total_best_gens = 0
        while(self.s >= fraud[self.n] and bestfit == 0): # stop when reached known optima (think about the environment!)
            #----------
            # Oerator registration
            #----------
            best = None
            if not self.dynamic and self.debug:
                print "using s: " + str(self.s)
            random.seed(time.time())
    
            # create an initial population of self.population individuals (where
            # each individual is a list of integers)
            pop = toolbox.population(n=self.population)
        
            NGEN = self.generations

            if self.debug:
                print("Start of evolution") 
        
            # Evaluate the entire population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        
            if self.debug:
                print("  Evaluated %i individuals" % len(pop))

            # Begin the evolution
            for g in range(self.generations):
                if self.debug:
                    print("-- Generation %i --" % g)

                for part in pop:
                    part.fitness.values = toolbox.evaluate(part)
                    if not part.best or part.best.fitness < part.fitness:
                        part.best = creator.Particle(part)
                        part.best.fitness.values = part.fitness.values
                    if not best or best.fitness < part.fitness:
                        best = creator.Particle(part)
                        best.fitness.values = part.fitness.values
                        if not total_best or total_best.fitness.values[0] >= best.fitness.values[0]:
                            total_best = creator.Particle(best)
                            total_best.fitness.values = best.fitness.values
                            total_best_gens = g + 1 # TODO: accumulate generations or only give generations for the s that found solution?
                for part in pop:
                    toolbox.update(part, best)

                # Gather all the fitnesses in one list and print the stats
                logbook.record(gen=g, evals=len(pop), **stats.compile(pop))

                # Gather all the fitnesses in one list and print the stats
                fits = [ind.fitness.values[0] for ind in pop]
            
                length = len(pop)
                mean = sum(fits) / length
                sum2 = sum(x*x for x in fits)
                std = abs(sum2 / length - mean**2)**0.5
                
                if self.debug:
                    print("  Min %s" % min(fits))
                    print("  Max %s" % max(fits))
                    print("  Avg %s" % mean)
                    print("  Std %s" % std)

                    print(" Best %s" % best)
                    print(" best fitness %s" % best.fitness.values[0])
                    print(" self.s %s" % self.s)
            
                    
                #prev_min = min(fits)
                if best.fitness.values == (0.0,):
                   best_ind = map(int, map(round, best))
                   best_s = self.s
                   best_found = best_ind
                   break
                
            if self.debug:
                print("-- End of (successful) evolution with s=" + str(self.s) + " --")
            if self.dynamic:
                break
            self.s -= 1
            if self.debug:
                print "self.s: " + str(self.s) + " fraud[self.n]: " + str(fraud[self.n])
        
        if best_ind == []:
            best_ind = map(int, map(round, best))
        self.s = best_s

        if self.dynamic:
            self.s = toolbox.sSize(best_ind)

        if self.debug:
            print("Best s is %s" % (self.s))
            print("Best individual is %s, %s" % (best_ind, best.fitness.values))
        
        total_ind = map(int,map(round,total_best))
        if self.debug:
            print("Total best individual is %s, %s" % (total_ind, total_best.fitness.values))
        f = open('out.txt', 'w')
        f.write('n: ' + str(self.n) + '\n')
        f.write('x: {')
        for i in reversed(xrange(2, 2*(self.n-self.ignoreSmallest), 2)):
            f.write(str(total_ind[i]) + ', ')
        f.write(str(total_ind[0]) + '}\ny: {')
        for i in reversed(xrange(3, 2*(self.n-self.ignoreSmallest), 2)):
            f.write(str(total_ind[i]) + ', ')
        f.write(str(total_ind[1]) + '}\ns: ' + str(int(best_s)))
        f.close()
        return {"bestS" : best_s,
                "bestInd" : total_ind,
                "bestFitness" : total_best.fitness.values,
                "nGens" : total_best_gens}

def main(argv):
    total = len(argv)
    cmdargs = str(argv)
    print ("The total numbers of args passed to the script: %d " % total)
    print ("Args list: %s " % cmdargs)
    # Parsing args one by one 
    n = 5
    if not argv:
        print ("n defaulting to "+str(n))
    else:
        print ("n: %s" % str(argv[0]))
        n = int(argv[0])
    
    pso = PSO(fitnessFun=costlierCostFloat, n=n, dynamic=False, debug=True, ignoreSmallest=True, fraudFlag=False, generations=20000, population=20)
    print pso.run()
        
if __name__ == "__main__":
    main(sys.argv[1:])
    
