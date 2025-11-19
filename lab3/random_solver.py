import numpy as np
from solver import Solver
from typing import Any, Callable, Dict, Tuple


class RandomSolver(Solver):
    """ A solver that draws n random points and returns the one with lowest goal function value"""
    def __init__(self, n: int = 256):
        self.n = n

    def get_parameters(self) -> Dict[str, Any]:
        return {'n': self.n}

    def solve(
            self, func: Callable[[np.ndarray], float], lower_bound: np.ndarray, upper_bound: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        bound_range = upper_bound - lower_bound
        points = lower_bound + np.random.rand(self.n, bound_range.shape[0]) * bound_range
        values = [func(point) for point in points]
        best = np.argmin(values)
        return points[best], values[best]


if __name__ == '__main__':
    def test_func(x):
        return np.sum(x ** 2)
    solver = RandomSolver()

    solutions = []
    values = []

    for i in range(25):
        solution, value = solver.solve(test_func, -np.ones(10), np.ones(10))
        solutions.append(solution)
        values.append(value)

        print(solution, value)

    print('Mean:', np.mean(values))
    print('Standard deviation:', np.std(values))
    print('Best solution overall', solutions[np.argmin(values)])
    print('Best solution value overall', np.min(values))
