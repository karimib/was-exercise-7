import math
from tsplib95 import loaders, distances
import numpy as np
import random

# Class representing the environment of the ant colony
"""
    rho: pheromone evaporation rate
"""


class Environment:
    def __init__(self, rho, number_of_ants):
        self.rho = rho
        self.number_of_ants = number_of_ants
        # Load the problem from the file
        self.problem = loaders.load('att48-specs/att48.tsp')
        self.problem_dimension = self.problem.dimension
        # Initialize the environment topology
        self.distance_matrix = np.loadtxt('att48-specs/att48_distance_matrix.txt')
        # Intialize the pheromone map in the environment
        self.nodes = list(self.problem.get_nodes())
        self.edges = list(self.problem.get_edges())
        self.pheromone_map = self.initialize_pheromone_map()

    def initialize_pheromone_map(self):

        curr = random.choice(self.nodes)
        visited = [curr]
        cost = 0

        while len(visited) < self.problem_dimension:
            min_node = -1
            min_dist = math.inf
            for node in self.nodes:
                if node not in visited:
                    print("curr: ", curr)
                    print("node: ", node)
                    dist = self.distance_matrix[curr - 1][node - 1]
                    if dist < min_dist:
                        min_dist = dist
                        min_node = node
            visited.append(min_node)
            cost += min_dist
            curr = min_node

        cost += self.distance_matrix[visited[-1]][visited[0]]

        tau = self.number_of_ants / cost
        #  Initialize the pheromone map with the tau value
        pheromone_map = np.array(tau * np.ones((self.problem_dimension, self.problem_dimension)))
        np.fill_diagonal(pheromone_map, 0)

        return pheromone_map

    # Update the pheromone trails in the environment
    def update_pheromone_map(self, ants):
        # Evaporate the pheromone trails
        self.pheromone_map = (1 - self.rho) * self.pheromone_map
        # Update the pheromone trails
        for ant in ants:
            deposit = 1 / ant.travelled_distance
            for edge in ant.visited_edges:
                self.pheromone_map[edge[0], edge[1]] += deposit
                self.pheromone_map[edge[1], edge[0]] += deposit

    # Get the pheromone trails in the environment
    def get_pheromone_map(self):
        return self.pheromone_map

    # Get the environment topology
    def get_possible_locations(self, start, visited):
        possible_locations = []
        for edge in self.edges:
            if edge[0] == start and edge[1] not in visited:
                possible_locations.append(edge[1])
        return possible_locations

    def get_distance(self, node1, node2):
        path = (node1, node2)
        return distances.pseudo_euclidean(*path)
