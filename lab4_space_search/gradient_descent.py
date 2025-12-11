# Viktoriia Nowotka
from numpy import cos, sin, pi, exp, sqrt, array, clip, linalg, copy
from solver import Solver
import numpy as np


class GradientDescent(Solver):
    def __init__(self, r_min, r_max, step, iterations, epsilon=1e-5):
        self.r_min = r_min
        self.r_max = r_max
        self.step = step
        self.iterations = iterations
        self.epsilon = epsilon
        self.stop_iter = None

    def get_parameters(self):
        return f"learning rate: {self.step}, stop iteration nr {self.stop_iter}"

    def gradient_ackley_1d(self, x, _):
        if np.abs(x) < self.epsilon:
            array([0.0, 0.0])

        df_dx = (2 * np.pi * np.exp(np.cos(2 * np.pi * x)) * np.abs(x) * np.sin(2 * np.pi * x) + 4 * x * np.exp(-np.abs(x) / 5.0))
        df_dy = 0.0
        return array([df_dx, df_dy])

    def gradient_ackley_2d(self, x, y):
        if np.abs(x) < self.epsilon and np.abs(y) < self.epsilon:
            return np.array([0.0, 0.0])

        df_dx_A = (2*x*np.sqrt(2)/(np.sqrt(x**2+y**2))) * np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
        df_dx_B = np.pi*sin(2*np.pi*x)*np.exp(0.5*(cos(2*np.pi*x) + cos(2*np.pi*y)))

        df_dy_A = (2*y*np.sqrt(2)/(np.sqrt(x**2+y**2))) * np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
        df_dy_B = np.pi*sin(2*np.pi*y)*np.exp(0.5*(cos(2*np.pi*x) + cos(2*np.pi*y)))

        df_dx = df_dx_A + df_dx_B
        df_dy = df_dy_A + df_dy_B

        if np.abs(x) < self.epsilon:
            df_dx = 0.0
        if np.abs(y) < self.epsilon:
            df_dy = 0.0

        return array([df_dx, df_dy])

    def solve(self, problem, x_start, *args):
        gradient_func = self._get_gradient(problem)

        x0 = array([x_start, args[0]], dtype=float)
        history = []

        for iteration in range(self.iterations):
            current_value = problem(x0[0], x0[1])
            history.append(current_value)

            d = gradient_func(x0[0], x0[1])

            gradient_norm = linalg.norm(d)
            if gradient_norm <= self.epsilon:
                self.stop_iter = iteration + 1
                break

            x0 -= self.step * d

        if self.stop_iter is None:
            self.stop_iter = self.iterations

        return x0, history

    def solve_with_trajectory(self, problem, x_start, *args):
        gradient_func = self._get_gradient(problem)

        x0 = array([x_start, args[0]], dtype=float)

        history_f = []
        history_x = []
        history_y = []

        for iteration in range(self.iterations):
            current_value = problem(x0[0], x0[1])
            history_f.append(current_value)
            history_x.append(x0[0])
            history_y.append(x0[1])

            d = gradient_func(x0[0], x0[1])
            x0 -= self.step * d

        return x0, history_f, array(history_x), array(history_y)

    def _get_gradient(self, problem):
        if problem.__name__ == "ackley_function_2d":
            gradient_func = self.gradient_ackley_2d
        elif problem.__name__ == "ackley_function":
            gradient_func = self.gradient_ackley_1d

        return gradient_func