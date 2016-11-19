# -*- coding: utf-8 -*-

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

# Force the placement of the biggest square or not by default
defaultForcePlacement = True
# Ignore the smallest square (1x1) or not by default
defaultIgnoreSmallest = True

# 3.2.3 Grid fit in report
def costlierCost(individual, tb):
    s = tb.s()
    squareSize = tb.squareSize
    grid = [[0 for _ in range(s)] for _ in range(s)]
    # Fill grid for all squares
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        
        for x_c in range(max(0,min(s,x_i)), min(s,x_i + squareSize(i))):
                for y_c in range(max(0,min(s,y_i)), min(s,y_i + squareSize(i))):
                        grid[x_c][y_c] = 1
        
    overlap_sum = 0
    overlapping_squares = []
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        # positive overflow
        x_non_overflow = max(0, min(x_i + squareSize(i), 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + squareSize(i), 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (squareSize(i))**2 - non_overflow_area
                
        # negative overflow
        if (x_i < 0 and y_i < 0):
            overflow_sum += abs(x_i) * squareSize(i) + abs(y_i) * squareSize(i) - abs(x_i)*abs(y_i)
        else:
            if (x_i < 0):
                overflow_sum += abs(x_i) * squareSize(i)
            if (y_i < 0):
                overflow_sum += abs(y_i) * squareSize(i)
        
        overlap_sum += 3*overflow_sum
        
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + squareSize(i), x_j + squareSize(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + squareSize(i), y_j + squareSize(j)) - max(y_i, y_j))
            overlap_area = x_overlap * y_overlap
            if overlap_area > 0:
                overlapping_squares.append(i)
            overlap_sum += 3*overlap_area
        
    
    # sum costs for getting out of overlap    
    for i in overlapping_squares:
        x_i = individual[i]
        y_i = individual[i+1]
        distances = [s]
        
        ## left direction
        pos = x_i
        steps = 0
        while (pos >= 0):
            # we might start outside the grid:
            if not (pos >= s):
                if grid[pos][min(s-1,max(y_i,0))] == 0:
                    break
            pos -= 1
            steps += 1
         
        # if we hit an edge we cant get out       
        if pos >= 0:
            distances.append(steps)
        
        ## right direction
        pos = x_i + squareSize(i)
        steps = 0
        while (pos <= s-1):
            # we might start outside the grid
            if not (pos < 0):
                if grid[pos][min(s-1,max(y_i,0))] == 0:
                    break
            pos += 1
            steps += 1
                
        # if we hit an edge we cant get out       
        if pos <= s-1:
            distances.append(steps)
            
        ## down direction
        pos = y_i
        steps = 0
        while (pos >= 0):
            # we might start outside the grid
            if not (pos >= s):
                if grid[min(s-1,max(x_i,0))][pos] == 0:
                    break
            pos -= 1
            steps += 1
                
        # if we hit an edge we cant get out       
        if pos >= 0:
            distances.append(steps)
            
        # up direction
        pos = y_i + squareSize(i)
        steps = 0
        while (pos <= s-1):
            # we might start outside the grid
            if not (pos < 0):
                if grid[min(s-1,max(x_i,0))][pos] == 0:
                    break
            pos += 1
            steps += 1
                
        # if we hit an edge we cant get out       
        if pos >= s-1:
            distances.append(steps)
        overlap_sum += min(distances)
                        
    return overlap_sum,

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

# list of provable smallest working s for different n values.
fraud = [0, 0, 3 , 5 , 7 , 9 , 11 , 13 , 15 , 18 , 21 , 24 , 27 , 30 , 33 , 36 , 39 , 43 , 47 , 50 , 54 , 58 , 62 , 66] 

# The square size of square i
def square_Size(i, ignoreSmallest=True):
    return (i/2) + 1 + ignoreSmallest

# Gives the side length of the enclosing square (to be used with dynamicArea)
def sSize(individual, toolbox=None):
    ignoreSmallest = toolbox.ignoreSmallest()
    squareSize = toolbox.squareSize
    min_h = 0; min_w = 0
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

# 3.2.1 Enumerative Overlap in report.
def numberOverlap(individual, tb):
    squareSize = tb.squareSize
    overlap_sum = 0
    s = tb.s()
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        x_non_overflow = max(0, min(x_i + squareSize(i), 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + squareSize(i), 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (squareSize(i))**2 - non_overflow_area
        overlap_sum += (overflow_sum > 0)
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + squareSize(i), x_j + squareSize(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + squareSize(i), y_j + squareSize(j)) - max(y_i, y_j))
            overlapArea = x_overlap * y_overlap
            overlap_sum += (overlapArea > 0)
    return overlap_sum,

# 3.2.1 Area Overlap in report.
def areaOverlap(individual, tb):
    squareSize = tb.squareSize
    overlap_sum = 0
    s = tb.s()
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        x_non_overflow = max(0, min(x_i + squareSize(i), 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + squareSize(i), 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (squareSize(i))**2 - non_overflow_area
        overlap_sum += overflow_sum
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + squareSize(i), x_j + squareSize(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + squareSize(i), y_j + squareSize(j)) - max(y_i, y_j))
            overlapArea = x_overlap * y_overlap
            overlap_sum += overlapArea
    return overlap_sum,

def dynamicArea(individual, tb):
    sSize = tb.sSize
    s = sSize(individual)
    n = tb.n()
    squareSize = tb.squareSize
    squarea = int(math.ceil(math.sqrt((n*(n+1)*(2*n+1)) / 6.0)))

    overlap_sum = 0
    for i in range(0,len(individual)-1,2):
        x_i = individual[i]
        y_i = individual[i+1]
        x_non_overflow = max(0, min(x_i + squareSize(i), 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + squareSize(i), 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (squareSize(i))**2 - non_overflow_area
        overlap_sum += overflow_sum
        for j in range(i+2,len(individual),2):
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + squareSize(i), x_j + squareSize(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + squareSize(i), y_j + squareSize(j)) - max(y_i, y_j))
            overlapArea = x_overlap * y_overlap
            overlap_sum += overlapArea

    deathroom = s**2 - squarea + overlap_sum
    deathroom_max = s**2 - squarea + n*(n+1)/2 
    return overlap_sum * (deathroom_max + 1) + deathroom,

# 3.2.1 Big Expensive in report.
def bigExpensive(individual, tb):
    overlap_sum = 0
    s = tb.s
    for i in range(0,len(individual)-1,2):
        outer_size = (squareSize(i))
        x_i = individual[i]
        y_i = individual[i+1]
        x_non_overflow = max(0, min(x_i + squareSize(i), 0 + s) - max(x_i, 0))
        y_non_overflow = max(0, min(y_i + squareSize(i), 0 + s) - max(y_i, 0))
        non_overflow_area = x_non_overflow * y_non_overflow
        overflow_sum = (squareSize(i))**2 - non_overflow_area
        overlap_sum += overflow_sum * outer_size
        for j in range(i+2,len(individual),2):
            inner_size = (squareSize(j)+1)
            x_j = individual[j]
            y_j = individual[j+1]
            x_overlap = max(0, min(x_i + squareSize(i), x_j + squareSize(j)) - max(x_i, x_j))
            y_overlap = max(0, min(y_i + squareSize(i), y_j + squareSize(j)) - max(y_i, y_j))
            overlapArea = x_overlap * y_overlap
            overlap_sum += overlapArea * max(outer_size, inner_size)
    return overlap_sum,

class GA:
    def defaultSelection(toolbox, self):
        return toolbox.register("select", tools.selTournament, tournsize=3)

    def defaultAttrCoord(toolbox, self):
        if self.forcePlacement:
            return toolbox.register("attrCoord", random.randint, self.n, self.s-2)
        else:
            return toolbox.register("attrCoord", random.randint, 0, self.s-2)

    def defaultEscapeSelection(toolbox, self):
        return toolbox.register("select", tools.selRandom)

    def defaultMutation(toolbox, self):
        if self.forcePlacement:
            return toolbox.register("mutate", mutation.mutRubber, sigma=(self.s/4), border=(self.s-2), indpb=0.15, lower_bound=(self.n))
        else:
            return toolbox.register("mutate", mutation.mutRubber, sigma=(self.s/4), border=(self.s-2), indpb=0.15)

    def defaultEscapeMutation(toolbox, self):
        return toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.s-2, indpb=0.45)

    def defaultCrossover(toolbox, self):
        toolbox.register("mate", tools.cxTwoPoint)

    def defaultStructureInit(toolbox, self):
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.attrCoord, (self.n-self.ignoreSmallest)*2)

    def __init__(self, escaper=False, dynamic=False, n=7, s=0, debug=False, ignoreSmallest=defaultIgnoreSmallest, forcePlacement=defaultForcePlacement, generations=100, population=2000, mutpb=0.7, cxpb=1.0, fitnessFun=areaOverlap, attrCoord=defaultAttrCoord, structureInit=defaultStructureInit, selection=defaultSelection, escapeSelection=defaultEscapeSelection, mutation=defaultMutation, escapeMutation=defaultEscapeMutation, crossover=defaultCrossover, fraudFlag=True, elitism=False, escapeShakes=15, escapeCount=10):
        self.escaper = escaper
        self.dynamic = dynamic
        self.n = n
        if s == 0:
            self.s = maximalS(self.n) if not fraudFlag else fraud[self.n]
        else:
            self.s = s
        self.debug = debug
        self.generations = generations
        self.population = population
        self.MUTPB = mutpb
        self.CXPB = cxpb

        self.fitnessFun = fitnessFun

        self.attrCoord = attrCoord
        self.structureInit = structureInit
        
        self.selection = selection
        self.escapeSelection = escapeSelection

        self.mutation = mutation
        self.escapeMutation = escapeMutation

        self.crossover = crossover

        self.ignoreSmallest = ignoreSmallest
        self.forcePlacement = forcePlacement

        self.fraud = fraudFlag

        self.elitism = elitism

        self.escapeShakes = escapeShakes
        self.escapeCount = escapeCount

    def run(self):
        if self.debug:
            print ("n:", self.n)
        self.s = maximalS(self.n) if not self.fraud else fraud[self.n]
        if self.debug:
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
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
    
        # Attribute generator 
        #                      define 'attr_bool' to be an attribute ('gene')
        #                      which corresponds to integers sampled uniformly
        #                      from the range [0,1] (i.e. 0 or 1 with equal
        #                      probability)
        self.attrCoord(toolbox, self)
    
        # Structure initializers
        #                         define 'individual' to be an individual
        #                         consisting of 100 'attr_bool' elements ('genes')
        self.structureInit(toolbox, self)

        # define the population to be a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
        

        # register the crossover operator
        self.crossover(toolbox, self)

        self.mutation(toolbox, self)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.selection(toolbox, self)
    
        best_found = None
        while(self.s >= fraud[self.n] and bestfit == 0): # stop when reached known optima (think about the environment!)
            #----------
            # Oerator registration
            #----------
            # register the goal / fitness function
            toolbox.register("evaluate", self.fitnessFun, tb=toolbox)

            if not self.dynamic and self.debug:
                print "using s: " + str(self.s)
            random.seed(time.time())
    
            # create an initial population of 300 individuals (where
            # each individual is a list of integers)
            #pop = toolbox.population(n=2500)
            pop = toolbox.population(n=self.population)
        
    
            # CXPB  is the probability with which two individuals
            #       are crossed
            #
            # MUTPB is the probability for mutating an individual
            #
            # NGEN  is the number of generations for which the
            #       evolution runs
            CXPB, MUTPB, NGEN = self.CXPB, self.MUTPB, self.generations

            if self.debug:
                print("Start of evolution") 
        
            # Evaluate the entire population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        
            if self.debug:
                print("  Evaluated %i individuals" % len(pop))
            prev_min = min([ind.fitness.values[0] for ind in pop])

            # Begin the evolution
            for i, g in enumerate(range(NGEN)):
                if self.debug:
                    print("-- Generation %i --" % g)

                if self.escaper:
                    if is_stuck:
                        shakes += 1
            
                        if shakes == self.escapeShakes:
                            stuck_counter = 0
                            self.selection(toolbox, self)

                            self.mutation(toolbox, self)

                            is_stuck = False
                            shakes = 0

                    if (stuck_counter == self.escapeCount):
                        backup_best = (tools.selBest(pop, 1)[0])
                        shakes = 0
                        self.escapeMutation(toolbox, self)

                        self.escapeSelection(toolbox, self)
                        is_stuck = True
                        stuck_counter = 0

                # Select the next generation individuals
                # Clone the selected individuals
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
        
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    # cross two individuals with probability CXPB
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
    
                        if self.forcePlacement:
                            child1[-1] = 0; child1[-2] = 0
                            child2[-1] = 0; child2[-2] = 0
                            # fitness values of the children
                            # must be recalculated later
                        del child1.fitness.values
                        del child2.fitness.values
            
                for mutant in offspring:
                    # mutate an individual with probability MUTPB
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        if self.forcePlacement:
                            mutant[-1] = 0; mutant[-2] = 0
                        del mutant.fitness.values
        
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
            
                if self.debug:
                    print("  Evaluated %i individuals" % len(invalid_ind))
            

                if self.elitism:
                    # Preserve the best individual from the previous generation
                    pop[:] = offspring + tools.selBest(pop, 1)
                else:
                    # The population is entirely replaced by the offspring
                    pop[:] = offspring
                    
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

                if min(fits) == prev_min:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
            
                prev_min = min(fits)
                if min(fits) == 0:
                    best_ind = (tools.selBest(pop, 1)[0])
                    best_s = self.s
                    best_found = best_ind
                    break
                
            bestfit = min(fits)
            if self.debug:
                print("-- End of (successful) evolution with s=" + str(self.s) + " --")
            if self.dynamic:
                break
            self.s -= 1
        
        if best_ind == []:
            best_ind = (tools.selBest(pop, 1)[0])
        self.s = best_s
        
        if self.dynamic:
            self.s = toolbox.sSize(best_ind)

        if self.debug:
            print("Best s is %s" % (self.s))
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            if backup_best and backup_best.fitness.values < best_ind.fitness.values:
                print("... but a better was found before escaping: %s, %s" % (backup_best, backup_best.fitness.values))
        f = open('out.txt', 'w')
        f.write('n: ' + str(self.n) + '\n')
        f.write('x: {')
        for i in reversed(xrange(2, 2*(self.n-self.ignoreSmallest), 2)):
            f.write(str(best_ind[i]) + ', ')
        f.write(str(best_ind[0]) + '}\ny: {')
        for i in reversed(xrange(3, 2*(self.n-self.ignoreSmallest), 2)):
            f.write(str(best_ind[i]) + ', ')
        f.write(str(best_ind[1]) + '}\ns: ' + str(int(self.s)))
        f.close()
        return {"bestS" : self.s,
                "bestInd" : best_ind,
                "backupBest" : backup_best if backup_best and backup_best.fitness.values < best_ind.fitness.values else None,
                "nGens" : g + 1}

def main(argv):
    total = len(argv)
    cmdargs = str(argv)
    print ("The total numbers of args passed to the script: %d " % total)
    print ("Args list: %s " % cmdargs)
    # Parsing args one by one 
    n = 10

    if not argv:
        print ("n defaulting to "+str(n))
    else:
        print ("n: %s" % str(argv[0]))
        n = int(argv[0])

    g = GA(debug=True, n=n, dynamic=False, fraudFlag=True, escaper=True, fitnessFun=areaOverlap)
    
    print g.run()
    
if __name__ == "__main__":
    main(sys.argv[1:])
