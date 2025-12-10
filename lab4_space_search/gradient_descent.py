# Viktoriia Nowotka
from solver import Solver

class GradientDescent(Solver):
    def __init__(self, r_min, r_max, step, iterations):
        self.r_min = r_min
        self.r_max = r_max
        self.step = step
        self.iterations = iterations
        self.stop_iter = None

    def get_parameters(self):
        return f"step: {self.step}, iterations: {self.iterations}"

    def solve(self, problem, x0, *args):
        pass