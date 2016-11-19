# Machine learning project
A machine learning project by myself [Mikael Östlund](https://github.com/eastlund), [Alexander Ek](https://github.com/aekh) and [Sebastian Rautila](https://github.com/SRautila).

For more information about the Square Packing Problem (SPP) and the results of this project, see project_report.pdf.

# Prerequisites
The code has been tested under `Python 2.7.6`.
You also need `numpy` installed for Python to run our code.

Since our project makes use of the [DEAP](https://github.com/DEAP/deap) package, make sure to follow their installation instructions before attempting to run our code.

# Usage
To run our best Genetic Algorithm (GA) configuration:
`python ga.py n`
Where n is the number of squares to pack (n > 2).

To run our best Particle Swarm Optimization (PSO) configuration:
`python pso.py n`
Where n is the number of squares to pack (n > 2).

The output to the shell will include the found best `s` (enclosing square side-length),  x- and y-positions of the squares as well as the fitness of the packing (0.0 if the packing is minimal and no squares overlap). After execution, the result will also be stored in a more viewable (and more easily-parsed) format in `out.txt`. Note that we ignore the 1x1 square by default, as we make sure that this can always fit inside our final enclosing square, which is why a packing of 5 squares can contain coordinates for only 4 squares. By default debug output is enabled to show how the generations evolve in each evolution step. Turn this off in `ga.py` and `pso.py` respectively if this is unwanted.

You can also run some experiments by executing
`python main.py`
which will run the experiments specified in `experiments.py`. However, we provide no parsing support or instructions for this in this readme.
