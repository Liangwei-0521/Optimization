import numpy as np


class GeneticAlgorithm:
    def __init__(self, length, population, num_variables, bound_left, bound_right, cross_rate, mutation_rate):
        self.length = length
        self.num_variables = num_variables
        self.population = population
        self.bound_left = bound_left
        self.bound_right = bound_right
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def decoding(self, population_scale):
        # 基因解码
        new = np.zeros((population_scale.shape[0], self.num_variables))
        for n_item in range(self.num_variables):
            # 比例系数
            alpha = population_scale[:, n_item * self.length:n_item * self.length + self.length].dot(
                2 ** np.arange(self.length)[::-1]) / float(2 ** self.length - 1)
            x_pop = alpha * (self.bound_right[n_item] - self.bound_left[n_item]) + self.bound_left[n_item]
            new[:, n_item] = x_pop

        return new

    def crossover(self, population_scale):
        # 基因交叉
        new_pop = []
        for member in population_scale:
            child = member
            if np.random.rand() < self.cross_rate:
                index = np.random.randint(self.population)
                mother = population_scale[index]
                # 交叉片段
                rand_point = np.random.randint(0, self.num_variables * self.length)
                child[rand_point:] = mother[rand_point:]
            # 基因变异
            self.mutation(child)
            new_pop.append(child)

        return new_pop

    def mutation(self, child):
        # 基因变异:片段内某个点发生突变
        if np.random.rand() < self.mutation_rate:
            index = np.random.randint(0, self.num_variables * self.length)
            child[index] = child[index] ^ 1

    def select(self, population, fitness):
        idx = np.random.choice(np.arange(self.population), size=self.population, replace=True,
                               p=fitness / (fitness.sum()))

        return population[idx,:]

    def information(self, fitness, optimal_pop):
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])
        new_value = GA.decoding(optimal_pop)
        print("最优的基因型：", optimal_pop[max_fitness_index])
        print("变量取值：", new_value[max_fitness_index])

    def main(self, ):
        pass


if __name__ == '__main__':
    # 目标函数
    def compute(variable):
        result = 100.0 * (variable[:, 1] - variable[:, 0] ** 2.0) ** 2.0 + (1 - variable[:, 0]) ** 2.0
        return result
    # 边界条件
    Bound_left = np.ones([1 * 2]) * -2.048
    Bound_right = np.ones([1 * 2]) * 2.048

    # 遗传算法
    GA = GeneticAlgorithm(length=24, population=80,
                          num_variables=2, bound_left=Bound_left, bound_right=Bound_right,
                          cross_rate=0.6, mutation_rate=0.01)
    # 迭代
    fitness = None
    new_pop = np.random.randint(low=0, high=2, size=(GA.population, GA.length * GA.num_variables))
    for iteration in range(500):
        # First:
        new_population = GA.crossover(new_pop)
        # Second:
        new_value = GA.decoding(np.array(new_population))
        # Third: 计算
        fitness = compute(new_value)
        # Fourth: 选择
        new_pop = GA.select(np.array(new_population), fitness)

    GA.information(fitness, new_pop)

