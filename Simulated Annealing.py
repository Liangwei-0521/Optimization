import numpy as np
import matplotlib.pyplot as plt


def target_function(x):

    return x[0] * x[1] + np.power(x[0] - x[1], 2)


def select(local_best):
    # select the best x
    for item in local_best['x']:
        value = target_function(item)
        local_best['y'].append(value)

    index = np.argmin(local_best['y'])
    return local_best['x'][index], local_best['y'][index]


class SA:
    def __init__(self, t_high, t_final, t_k, iter, bound):

        self.iter = iter
        self.bound = [-bound, bound]
        self.last_value = None
        self.t_alpha = t_k
        self.temperature_high = t_high
        self.temperature_final = t_final
        self.T = self.temperature_high

    def generate_new_x(self, x):
        # disturb the current x
        while True:
            x = list(map(lambda a: a + (np.random.rand() - np.random.rand()), x))
            for a in x:
                if a in self.bound:
                    pass
                else:
                    break
            return x

    def metropolis(self, x):
        value = target_function(x)
        delta_e = value - self.last_value
        if delta_e < 0:
            return 1
        else:
            '''
            # The higher the temperature,
             the more likely it is to accept the bad solution 
             (which helps to jump out of the local optimal solution)
            '''

            e = np.exp(-delta_e / self.T)
            if np.random.randn() < e:
                return 1
            else:
                return 0

    def run(self, ):
        # generate the initial x
        x = [np.random.rand(), np.random.rand()]
        global_best = {
            'x': [],
            'y': [],
        }

        while self.T >= self.temperature_final:
            local_best = {
                'x': [],
                'y': []
            }
            local_best['x'].append(x)

            for time in range(self.iter):
                self.last_value = target_function(x)
                # generate the new x
                x_ = self.generate_new_x(x)
                if self.metropolis(x_) == 1:
                    x = x_
                    local_best['x'].append(x)

            self.T = self.t_alpha * self.T
            iter_x, iter_y = select(local_best)

            global_best['x'].append(iter_x)
        # output the best answer
        best_answer_x, best_answer_y = select(global_best)

        plt.plot(range(len(global_best['y'])), global_best['y'])
        plt.show()
        print('Best the x : ', best_answer_x, '\n', 'Best the value:', best_answer_y)


if __name__ == '__main__':
    sa = SA(t_high=100, t_final=0.01, t_k=0.99, iter=100, bound=5)
    sa.run()
