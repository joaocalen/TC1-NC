import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error


# Activations
def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


# Neuiral Network
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the neural network
        :param input_size: number of input neurons
        :param hidden_size: number of hidden neurons
        :param output_size: number of output neurons
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # # Initialize weights and biases using Xavier initialization
        # self.v = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        # self.v0 = np.zeros(hidden_size)
        # self.w = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))
        # self.w0 = np.zeros(output_size)

        # Initialize weights and biases
        self.v = np.random.rand(hidden_size, input_size)
        self.v0 = np.random.rand(hidden_size)
        self.w = np.random.rand(output_size, hidden_size)
        self.w0 = np.random.rand(output_size)

    def predict(self, phi: np.ndarray) -> np.ndarray:
        """
        Make a prediction with the neural network
        :param phi: input data
        :return: prediction
        """
        hidden_input = np.dot(self.v, phi) + self.v0
        hidden_output = relu(hidden_input)
        output = np.dot(self.w, hidden_output) + self.w0
        return linear(output)

    def get_weights(self):
        return np.concatenate([self.v.flatten(), self.v0, self.w.flatten(), self.w0])

    def set_weights(self, weights):
        v_end = self.hidden_size * self.input_size
        v0_end = v_end + self.hidden_size
        w_end = v0_end + (self.output_size * self.hidden_size)

        self.v = weights[:v_end].reshape(self.hidden_size, self.input_size)
        self.v0 = weights[v_end:v0_end]
        self.w = weights[v0_end:w_end].reshape(self.output_size, self.hidden_size)
        self.w0 = weights[w_end:]


class PSO:
    def __init__(self, network, data, targets, num_particles, max_iter, w=0.5, c1=2.0, c2=2.0):
        """
        Initialize the Particle Swarm Optimization algorithm
        :param network: neural network
        :param data: input data
        :param targets: target data
        :param num_particles: number of particles
        :param max_iter: maximum number of iterations
        :param w: inertia weight
        :param c1: cognitive weight
        :param c2: social weight
        """
        self.network = network
        self.data = data
        self.targets = targets
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.dim = len(network.get_weights())
        self.swarm = np.random.rand(num_particles, self.dim)
        self.velocity = np.zeros((num_particles, self.dim))
        self.pbest = self.swarm.copy()
        self.pbest_fitness = np.array([self.evaluate_fitness(weights) for weights in self.swarm])
        self.best_weights = self.pbest[np.argmin(self.pbest_fitness)].copy()
        self.best_weights_fitness = np.min(self.pbest_fitness)
        self.lbest = self.swarm.copy()
        self.lbest_fitness = self.pbest_fitness.copy()
        self.history = []

    def evaluate_fitness(self, weights):
        self.network.set_weights(weights)
        return fitness(self.network, self.data, self.targets)

    def calculate_diversity(self):
        return np.mean(np.linalg.norm(self.swarm - np.mean(self.swarm, axis=0), axis=1))

    def adaptive_neighborhood(self, diversity):
        neighborhood_size = min(
            max(2, int(self.num_particles * (1 - diversity))), self.num_particles
        )
        for i in range(self.num_particles):
            distances = np.linalg.norm(self.swarm - self.swarm[i], axis=1)
            sorted_indices = np.argsort(distances)
            neighborhood_indices = sorted_indices[:neighborhood_size]
            best_neighbor_index = np.argmin(self.pbest_fitness[neighborhood_indices])
            self.lbest[i] = self.pbest[neighborhood_indices[best_neighbor_index]]
            self.lbest_fitness[i] = self.pbest_fitness[
                neighborhood_indices[best_neighbor_index]
            ]

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.num_particles):
                self.network.set_weights(self.swarm[i])
                current_fitness = fitness(self.network, self.data, self.targets)

                # Update personal best
                if current_fitness < self.pbest_fitness[i]:
                    self.pbest[i] = self.swarm[i].copy()
                    self.pbest_fitness[i] = current_fitness

                # Update global best
                if current_fitness < self.best_weights_fitness:
                    self.best_weights = self.swarm[i].copy()
                    self.best_weights_fitness = current_fitness

            # Calculate swarm diversity
            diversity = self.calculate_diversity()

            # Adaptive neighborhood adjustment
            self.adaptive_neighborhood(diversity)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = (
                    self.w * self.velocity[i]
                    + self.c1 * r1 * (self.pbest[i] - self.swarm[i])
                    + self.c2 * r2 * (self.lbest[i] - self.swarm[i])
                )
                self.swarm[i] += self.velocity[i]

            # Update the best weights found
            self.network.set_weights(self.best_weights)
            best_rmse = fitness(self.network, self.data, self.targets)
            print(f"Iteration {t+1}/{self.max_iter} - RMSE: {best_rmse}", end="\r")
            self.history.append(best_rmse)

# PSO fitness function
def fitness(network, data, targets):
    predictions = np.array([network.predict(phi) for phi in data])
    rmse = root_mean_squared_error(targets, predictions)
    return rmse
