#Viktoriia Nowotka
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import matplotlib.pyplot as plt


def ackley_function(x, y):
 return -20.0 * exp(-0.2 * sqrt(x**2))-exp(cos(2 * pi * x)) + 20 + e


def two_dimensional_ackley_function(x, y):
 return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + 20 + e

def main():
    r_min, r_max = -30, 30

    plt.subplots(1, 3, subplot_kw={'projection': '3d'})

    xaxis = arange(r_min, r_max, 2.0)
    yaxis = arange(r_min, r_max, 2.0)
    x, y = meshgrid(xaxis, yaxis)

    results = two_dimensional_ackley_function(x, y)

    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet', shade="false")
    plt.show()


if __name__ == '__main__':
    main()