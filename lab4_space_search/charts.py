# Viktoriia Nowotka
from numpy import exp, sqrt, cos, e, pi
import matplotlib.pyplot as plt
import numpy as np

# 1D
def ackley_function(x, y):
    return -20.0 * exp(-0.2 * sqrt(x**2)) - exp(cos(2 * pi * x)) + 20 + e


def visualization_ackley_function(r_min, r_max):
    inputs = np.arange(r_min, r_max, 1)
    outputs = np.array([ackley_function(x, 0) for x in inputs])

    plt.figure(figsize=(12, 7))
    plt.plot(inputs, outputs, color='blue', linewidth=2)
    plt.title("Wykres 1. Funkcji Ackleya")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    plt.show()


def visualization_convergence_ackley_function(history):
    plt.figure(figsize=(12, 7))
    for step, hist in history.items():
        plt.plot(hist, label=f'Krok (Step) = {step}')

    plt.title('Wykres 2. Zbieżność Gradient Descent dla funkcji Ackleya')
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość f(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def visualize_best_step_2d(final_pos, history_f, path_x, optimal_step, r_vis=5):
    x_range = np.arange(-r_vis, r_vis, 0.01)
    f_range = np.vectorize(ackley_function)(x_range, x_range)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_range, f_range, color='blue', linewidth=1, label='Funkcja f(x)')
    ax.plot(path_x, history_f, color='red', marker='o', markersize=5, linewidth=2, label='Trajektoria')

    ax.scatter(path_x[0], history_f[0], color='green', s=100, zorder=5, label='Punkt początkowy')
    ax.scatter(final_pos[0], history_f[-1], color='cyan', s=100, zorder=5, label='Punkt końcowy')

    plt.title("Wykres 3. Trajektoria minimalizacji", fontsize=16)
    ax.set_title(f'Trajektoria dla optymalnego kroku {optimal_step}')
    ax.set_xlabel('X')
    ax.set_ylabel('Wartość funkcji f(x)')
    ax.grid(True)
    plt.legend()
    plt.show()

# 2D
def ackley_function_2d(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.e


def visualization_ackley_function_3d(r_min, r_max):
    xaxis = np.arange(r_min, r_max, 1)
    yaxis = np.arange(r_min, r_max, 1)

    X, Y = np.meshgrid(xaxis, yaxis)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ackley_function_2d(X[i, j], Y[i, j])

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=False)

    fig.colorbar(surface, shrink=0.5, aspect=5, label='Wartość funkcji')

    ax.set_title('Wykres 4. Funkcji Ackleya 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wartość Funkcji f(x, y)')
    ax.view_init(elev=35, azim=-120)
    plt.show()




