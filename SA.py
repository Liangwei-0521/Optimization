import math
from random import random
import matplotlib.pyplot as plt


def func(x, y):  # 函数优化问题
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


# x为公式里的x1,y为公式里面的x2
class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.iter = iter  # 内循环迭代次数,即为L =100
        self.alpha = alpha  # 降温系数，alpha=0.99
        self.T0 = T0  # 初始温度T0为100
        self.Tf = Tf  # 温度终值Tf为0.01
        self.T = T0  # 当前温度
        self.x = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个x的值
        self.y = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个y的值
        self.most_best = []
        """
        random()这个函数取0到1之间的小数
        如果你要取0-10之间的整数（包括0和10）就写成 (int)random()*11就可以了，11乘以零点多的数最大是10点多，最小是0点多
        该实例中x1和x2的绝对值不超过5（包含整数5和-5），（random() * 11 -5）的结果是-6到6之间的任意值（不包括-6和6）
        （random() * 10 -5）的结果是-5到5之间的任意值（不包括-5和5），所有先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
        """
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):  # 扰动产生新解的过程
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break  # 重复得到新解，直到产生的新解满足约束条件
        return x_new, y_new

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):  # 获取最优目标函数值
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        count = 0
        # 外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:

            # 内循环迭代100次
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])  # f为迭代一次后的值
                x_new, y_new = self.generate_new(self.x[i], self.y[i])  # 产生新解
                f_new = self.func(x_new, y_new)  # 产生新值
                if self.Metrospolis(f, f_new):  # 判断是否接受新值
                    self.x[i] = x_new  # 如果接受新值，则把新值的x,y存入x数组和y数组
                    self.y[i] = y_new
            # 迭代L次记录在该温度下最优解
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            count += 1

            # 得到最优解
        f_best, idx = self.best()
        print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")


sa = SA(func)
sa.run()

plt.plot(sa.history['T'], sa.history['f'])
plt.title('SA')
plt.xlabel('T')
plt.ylabel('f')
plt.gca().invert_xaxis()
plt.show()

