import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    return x[0] * x[1] + np.power(x[0] + x[1], 2)


class PSO:
    def __init__(self, w, c_1, c_2, iteration, population, dimension):
        self.w = w
        self.c_1 = c_1
        self.c_2 = c_2
        self.r_1 = np.random.randn()
        self.r_2 = np.random.randn()
        self.iter = iteration
        self.dimension = dimension
        self.population = population
        self.x = np.zeros(shape=(self.population, self.dimension))
        self.v = np.zeros(shape=(self.population, self.dimension))
        # 全局粒子的最优解
        self.global_best_x = np.zeros(shape=(1, self.dimension))
        # 每个粒子的最优解
        self.local_best_x = np.zeros(shape=(self.population, self.dimension))
        # 历史最优值
        self.local_fitness = np.zeros(shape=(1, self.population))
        self.global_fitness = None

    def init_population(self, ):
        for i in range(self.population):
            for j in range(self.dimension):
                self.x[i, j] = np.random.randn()
                self.v[i, j] = np.random.randn()
            self.local_best_x[i] = self.x[i]
            self.local_fitness[0, i] = target_function(self.x[i])

        index = np.argmin(self.local_fitness, axis=1)
        self.global_best_x = self.local_best_x[index]

        self.global_fitness = target_function(self.global_best_x[0])
        print(' initial optimal x', self.global_best_x[0],
              '\n initial fitness value:', self.global_fitness)

    def run(self, ):
        all_fitness = []
        for i in range(self.iter):

            for k in range(self.population):
                # update the speed of Particle
                self.v[k] = (self.w * self.v[k] +
                             self.c_1 * self.r_1 * (self.local_best_x[k] - self.x[k]) +
                             self.c_2 * self.r_2 * (self.global_best_x[0] - self.x[k]))
                # update the position of Particle
                self.x[k] = self.x[k] + self.v[k]

            for j in range(self.population):
                temp = target_function(self.x[j])

                if temp < self.local_fitness[0, j]:
                    # update the optimal each particle
                    self.local_fitness[0, j] = temp
                    self.local_best_x[j] = self.x[j]
                    optional_x = self.local_best_x[np.argmin(self.local_fitness, axis=1)]

                    if self.global_fitness > target_function(optional_x[0]):
                        self.global_best_x = optional_x
                        self.global_fitness = target_function(optional_x[0])
                        all_fitness.append(self.global_fitness)

        print('--- end the iteration ---')
        print(' optimal x : ', self.global_best_x, ' \n optimal value : ', self.global_fitness)

        return all_fitness


if __name__ == '__main__':
    pso = PSO(w=0.3, c_1=1.6, c_2=2.0, iteration=100, population=50, dimension=2)
    pso.init_population()
    fitness = pso.run()

    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.plot(range(0, len(fitness)), fitness)
    plt.show()
