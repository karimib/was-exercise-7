import numpy as np

# Class representing an artificial ant of the ant colony
"""
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
"""


class Ant():
    def __init__(self, alpha: float, beta: float, initial_location):
        self.alpha = alpha
        self.beta = beta
        self.current_location = initial_location
        self.travelled_distance = 0
        self.visited_edges = []
        self.environment = None

    # The ant runs to visit all the possible locations of the environment 
    def run(self):
        self.visited_edges.append(self.current_location)
        while len(self.visited_edges) < self.environment.problem_dimension:
            next_location = self.select_path()
            self.travelled_distance += self.get_distance(self.current_location, next_location)
            self.current_location = next_location
            self.visited_edges.append(self.current_location)

    def calculate_probability(self):
        reachable_nodes = self.environment.get_possible_locations(self.current_location, self.visited_edges)
        list_of_probabilities = []

        for node in reachable_nodes:
            tau_nij = self.tau_nij(self.current_location, node)
            list_of_probabilities.append(tau_nij)

        total = sum(list_of_probabilities)
        probability = [x / total for x in list_of_probabilities]
        return probability

    def tau_nij(self, start, end):
        tau = self.environment.pheromone_map[start - 1][end - 1]
        n_ij = 1 / self.environment.distance_matrix[start - 1][end - 1]
        return tau ** self.alpha * n_ij ** self.beta

    # Select the next path based on the random proportional rule of the ACO algorithm
    def select_path(self):
        reachable_nodes = self.environment.get_possible_locations(self.current_location, self.visited_edges)
        probability = self.calculate_probability()
        return np.random.choice(reachable_nodes, p=probability)

    def get_distance(self, start, end):
        return self.environment.get_distance(start, end)

    # Position an ant in an environment
    def join(self, environment):
        self.environment = environment
