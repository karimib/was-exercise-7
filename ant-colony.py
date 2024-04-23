import random

import numpy as np

from environment import Environment
from ant import Ant

# Class representing the ant colony
"""
    ant_population: the number of ants in the ant colony
    iterations: the number of iterations 
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
    rho: pheromone evaporation rate
"""


class AntColony:
    def __init__(self, ant_population: int, iterations: int, alpha: float, beta: float, rho: float):
        self.ant_population = ant_population
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        # Initialize the environment of the ant colony
        self.environment = Environment(self.rho, self.ant_population)
        # Initilize the list of ants of the ant colony
        self.ants = []

        # Initialize the ants of the ant colony
        for i in range(ant_population):
            # Initialize an ant on a random initial location 
            ant = Ant(self.alpha, self.beta, random.choice(range(self.environment.problem_dimension)))
            # Position the ant in the environment of the ant colony so that it can move around
            ant.join(self.environment)
            # Add the ant to the ant colony
            self.ants.append(ant)

    # Solve the ant colony optimization problem  
    def solve(self):
        # The solution will be a list of the visited cities
        solution = []
        # Initially, the shortest distance is set to infinite
        shortest_distance = np.inf

        for i in range(self.iterations):
            for ant in self.ants:
                ant.run()
                self.environment.update_pheromone_map(ant)
                if ant.travelled_distance < shortest_distance:
                    shortest_distance = ant.travelled_distance
                    solution = ant.visited_edges

        return solution, shortest_distance


def main():
    no_ants = 48
    iterations = 50
    alpha = 2
    beta = 5
    rho = 0.2

    ant_colony = AntColony(no_ants, iterations, alpha, beta, rho)
    solution, distance = ant_colony.solve()

    print("Solution: ", solution)
    print("Solution: ", distance)


if __name__ == '__main__':
    main()
