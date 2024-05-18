import math
import random
import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, low, up, size, n_steps, dimension) -> None:
        self.w = 0.7
        self.c1 = 2
        self.c2 = 2
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)
        # the number of particle
        self.size = size
        # the steps of iteration
        self.n_steps = n_steps
        # the dimension of variable
        self.dimension = dimension
        # position
        self.x = np.zeros(shape=(self.size, self.dimension))
        # speed
        self.v = np.zeros(shape=(self.size, self.dimension))
        # the best position of local
        self.local_best_x = np.zeros(shape=(self.size, self.dimension))
        # the best position of global
        self.global_best_x = np.zeros(shape=(self.size, self.dimension))

    def calculate_fitness(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        y = math.floor((x2 * np.exp(x1) + x3 * np.sin(x2) + x4 + x5) * 100) / 100
        return y

    def init_population(self, ):
        # initialized state
        temp = -1e5
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)

            self.local_best_x[i] = self.x[i]
            fit = self.calculate_fitness(self.local_best_x)
            # update
            if fit > temp:
                self.global_best_x = self.local_best_x
                temp = fit

    def iterator(self, x):
        # update
        for i in range(self.n_steps):
            # update the v (speed)
            self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (self.local_best_x - self.x[i]) \
                        + self.c2 * random.uniform(0, 1) * (self.global_best_x - self.x[i])

            # limit the v (speed)
            pass

            # update the position
            self.x = self.x[i] + self.v[i]

            # limit the x (position)
            pass

            # update the local_best_x and the global_best_x
            if self.calculate_fitness(self.x[i]) > self.calculate_fitness(self.local_best_x[i]):
                self.global_best_x[i] = self.x[i]

            if self.calculate_fitness(self.x[i]) > self.calculate_fitness(self.global_best_x):
                self.global_best_x = self.x[i]

    def process(self, ):
        best = []
        self.final_best = np.array([1, 2, 3, 4, 5])
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print('当前最佳位置：{}'.format(self.final_best))
            temp = self.fitness(self.final_best)
            print('当前的最佳适应度：{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        plt.figure()
        plt.grid(axis='both')
        plt.plot(t, best, color='red', marker='.', ms=10)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数")  # X轴标签
        plt.ylabel(u"适应度")  # Y轴标签
        plt.title(u"迭代过程")  # 标题
        plt.show()


if __name__ == '__main__':
    pso_algorithm = PSO()
