import numpy as np
import matplotlib.pyplot as plt

class AntColony:
    

    def __init__(self, distances, coordinates, n_ants=20, n_iterations=200, alpha=1, beta=2, evaporation=0.5, Q=1):
      
        self.distances = distances
        self.coordinates = coordinates
        self.n_cities = distances.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.pheromones = np.ones((self.n_cities, self.n_cities))

    def run(self):
       
        best_path = None
        best_length = float("inf")

        for _ in range(self.n_iterations):
            paths, lengths = self._construct_solutions()
            self._update_pheromones(paths, lengths)

            min_length = min(lengths)
            if min_length < best_length:
                best_length = min_length
                best_path = paths[np.argmin(lengths)]

        return best_path, best_length

    def _construct_solutions(self):
        #for listing paths and its cost/lenghtss
       
        paths = []
        lengths = []

        for _ in range(self.n_ants):
            path = self._construct_path()
            length = self._path_length(path)
            paths.append(path)
            lengths.append(length)

        return paths, lengths

    def _construct_path(self):
        
        path = [0] 
        for _ in range(self.n_cities - 1):
            current_city = path[-1]
            next_city = self._select_next_city(current_city, path)
            path.append(next_city)

        return path + [0]  

    def _select_next_city(self, current_city, path):
       #selecting next city based on pheromone no : 
        unvisited = list(set(range(self.n_cities)) - set(path))
        pheromone = self.pheromones[current_city, unvisited]
        heuristic = 1 / self.distances[current_city, unvisited]

        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities /= probabilities.sum()

        return np.random.choice(unvisited, p=probabilities)

    def _path_length(self, path):
     
        return sum(self.distances[path[i], path[i + 1]] for i in range(len(path) - 1))

    def _update_pheromones(self, paths, lengths):
       #updating pheromone when the path used 
        self.pheromones *= (1 - self.evaporation)

        for path, length in zip(paths, lengths):
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i + 1]] += self.Q / length
                self.pheromones[path[i + 1], path[i]] += self.Q / length

    def visualize(self, best_path):
       
        plt.figure(figsize=(8, 6))

        for i, (x, y) in enumerate(self.coordinates):
            plt.scatter(x, y, c='red', marker='o', s=100)
            plt.text(x, y, f'  {i}', fontsize=12, verticalalignment='bottom')

        for i in range(len(best_path) - 1):
            x1, y1 = self.coordinates[best_path[i]]
            x2, y2 = self.coordinates[best_path[i + 1]]
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

        plt.title("Ant Colony Optimization - Best Path (5 Cities)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)

    coordinates = np.random.randint(0, 20, size=(4, 2))
    coordinates = np.vstack([[0,0], coordinates]) 

    n = len(coordinates)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

    colony = AntColony(distances, coordinates, n_ants=20, n_iterations=200)
    best_path, best_length = colony.run()

    print("Best Path:", best_path)
    print("Best Path Length:", best_length)

    colony.visualize(best_path)