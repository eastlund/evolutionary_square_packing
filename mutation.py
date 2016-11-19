import random
import copy

from itertools import repeat
from collections import Sequence

# Mutator with "rubber-like" behaviour. Doesn't go out of bounds, moves in a gaussian way.
# 3.3.1 Local Gaussian Mutation in report.
def mutRubber(individual, sigma, border, indpb, lower_bound=0):
    size = len(individual)
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
    
    for i, s in zip(xrange(size), sigma):
        if random.random() < indpb:
            if (lower_bound != 0):
                if (i % 2 == 0):
                    other_coord = i + 1
                else:
                    other_coord = i - 1

                if individual[other_coord] < lower_bound:
                    individual[i] = int(round(min(border, max(lower_bound, random.gauss(individual[i], s)))))
                else:
                    individual[i] = int(round(min(border, max(0, random.gauss(individual[i], s)))))
                
            else:
                individual[i] = int(round(min(border, max(0, random.gauss(individual[i], s)))))
    
    return individual,

