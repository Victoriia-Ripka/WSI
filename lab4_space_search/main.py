# Viktoriia Nowotka
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import GradientDescent


def ackley_function(x, y):
 return -20.0 * exp(-0.2 * sqrt(x**2))-exp(cos(2 * pi * x)) + 20 + e


def two_dimensional_ackley_function(x, y):
 return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + 20 + e


def main():
    r_min, r_max = -30, 30
    max_iter = 1000
    # zbadać wpływ rozmiaru kroku na działanie algorytmu
    steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # zbadać wpływ punktu początkowego
    x0 = 0
    y0 = 0

    print("For one-dimensional function")
    for step in steps:
        gradient_descent = GradientDescent(r_min, r_max, step, max_iter)
        result = gradient_descent.solve(ackley_function, x0)
        print(f"step: {step}, result: {result}")

    print("For two-dimensional function")
    for step in steps:
        gradient_descent = GradientDescent(r_min, r_max, step, max_iter)
        result = gradient_descent.solve(two_dimensional_ackley_function, x0, y0)
        print(f"step: {step}, result: {result}")

    # plt.subplots(1, 3, subplot_kw={'projection': '3d'})
    #
    # xaxis = arange(r_min, r_max, 2.0)
    # yaxis = arange(r_min, r_max, 2.0)
    # x, y = meshgrid(xaxis, yaxis)
    #
    # results = two_dimensional_ackley_function(x, y)
    #
    # figure = plt.figure()
    # axis = figure.add_subplot(projection='3d')
    # axis.plot_surface(x, y, results, cmap='jet', shade="false")
    # plt.show()


if __name__ == '__main__':
    main()