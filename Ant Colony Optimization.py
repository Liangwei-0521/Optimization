import numpy as np


def target_function(x):
    return np.power(x[0] - x[1], 2) - np.power(1 - x[1] * x[1], 2) - 3 * np.power(1 + x[1], 2)


class ACO:
    def __init__(self, m, x_upper, x_lower, n_variables, max_iter):
        self.m = m
        self.x_upper = x_upper
        self.x_lower = x_lower
        self.n_variables = n_variables
        self.max_iter = max_iter
        # value matrix
        self.tau = np.zeros(shape=(1, m))
        # position matrix
        self.position_m = np.zeros(shape=(self.m, self.n_variables))
        # transition
        self.prob_transition = np.zeros(shape=(self.max_iter, self.m))
        self.rho = 0.9
        self.prob = 0.2
        self.step = 0.05
        self.best_tau = None

    def init_ant_colony(self, ):
        for i in range(self.m):
            for j in range(self.n_variables):
                self.position_m[i][j] = np.random.uniform(self.x_lower, self.x_upper)
            self.tau[0, i] = target_function(self.position_m[i, :])
        # initial best tau
        index = np.argmin(self.tau, axis=1)
        self.best_tau = self.tau[0, index]

    def iteration(self, ):
        for iter in range(1, self.max_iter):
            # update probability
            for j in range(self.m):
                self.prob_transition[iter, j] = (self.best_tau - self.tau[0, j]) / self.best_tau

            # update the position
            for k in range(self.m):
                if self.prob_transition[iter, k] < self.prob:
                    # local search
                    temp = self.position_m[k, :] + self.step * (1 / iter) * (2 * np.random.rand() - 1)
                    temp = np.clip(temp, self.x_lower, self.x_upper)

                else:
                    # global search
                    temp = self.position_m[k, :] + (self.x_upper - self.x_lower) * (np.random.randn() - 0.5)
                    temp = np.clip(temp, self.x_lower, self.x_upper)

                # determine whether move or not
                if target_function(temp) < self.tau[0, k]:
                    self.position_m[k, :] = temp

                # update the tau
                self.tau[0, k] = (1 - self.rho) * self.tau[0, k] + target_function(self.position_m[k, :])

            # iteration information
            index = np.argmin(self.tau, axis=1)
            x = self.position_m[index, :][0]
            value = self.tau[0, index]
            print(' the iteration is : {}, the x is {}, the value is {}'.format(iter, x, value))

        best_tau_index = np.argmin(self.tau, axis=1)
        best_value = self.tau[0, best_tau_index]
        best_position = self.position_m[best_tau_index, :]
        print(' Optimal x is: {}, \n Best value is: {} '.format(best_position[0], best_value))


if __name__ == '__main__':
    aco = ACO(m=300, x_upper=-10, x_lower=50, n_variables=2, max_iter=100 + 1)
    aco.init_ant_colony()
    aco.iteration()
