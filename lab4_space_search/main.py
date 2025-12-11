# Viktoriia Nowotka
from numpy import exp, sqrt, cos, e, pi, meshgrid, arange
import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import GradientDescent
from charts import *


def ackley_function_2d(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.e


def visualization_ackley_function_3d(r_min, r_max):
    xaxis = np.arange(-r_min, r_max, 1)
    yaxis = np.arange(-r_min, r_max, 1)

    X, Y = np.meshgrid(xaxis, yaxis)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ackley_function_2d(X[i, j], Y[i, j])
    print(Z)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=False)

    fig.colorbar(surface, shrink=0.5, aspect=5, label='Wartość funkcji')

    ax.set_title('Wykres 3. Funkcji Ackleya 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wartość Funkcji f(x, y)')
    ax.view_init(elev=45, azim=-120)
    plt.show()


def visualization_convergence_ackley_function_2d(history):
    plt.figure(figsize=(12, 7))
    for step, hist in history.items():
        # print(step, hist)
        plt.plot(hist, label=f'Krok (Step) = {step}')

    plt.title('Zbieżność Gradient Descent dla funkcji Ackleya 2d')
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość Funkcji Celu $f(x, y)$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 10)  # Ustawienie limitu dla lepszej wizualizacji zbieżności
    plt.show()


def visualize_best_step_3d(final_pos, history_f, path_x, path_y, optimal_step):
    r_vis = 5
    xaxis = np.arange(-r_vis, r_vis, 0.1)
    yaxis = np.arange(-r_vis, r_vis, 0.1)
    X, Y = np.meshgrid(xaxis, yaxis)
    Z = np.vectorize(ackley_function_2d)(X, Y)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, label='Powierzchnia Ackleya')

    ax.plot(path_x, path_y, history_f, color='red', marker='o', markersize=3, linewidth=2)

    ax.scatter(path_x[0], path_y[0], history_f[0], color='green', s=50, label='Start')
    ax.scatter(final_pos[0], final_pos[1], history_f[-1], color='cyan', s=50, label='Stop')

    ax.set_title(f'Trajektoria dla optymalnego kroku {optimal_step}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wartość Funkcji f(x,y)')
    ax.view_init(elev=50, azim=-120)
    ax.plot([], [], [], color='red', linewidth=2)
    plt.legend()
    plt.show()


def main():
    r_min, r_max = -32.768, 32.768
    max_iter = 200
    # TODO zbadać wpływ rozmiaru kroku na działanie algorytmu
    steps = [0.01, 0.02, 0.05, 0.075, 0.1] # 0.005, 0.01, 0.015, 0.02, 0.05, 0.075, 0.09, 0.1, 0.15, 0.2, 0.25
    # TODO zbadać wpływ punktu początkowego
    x0 = 23.0
    y0 = 0.0

    history_1d = {}
    final_values_1d = []

    # 1D Wykres funkcji
    visualization_ackley_function(r_min, r_max)

    # print("Zbieranie danych dla ackley_function...")
    for step in steps:
        gradient_descent = GradientDescent(r_min, r_max, step, max_iter)
        x, history = gradient_descent.solve(ackley_function, x0, y0)
        history_1d[step] = history
        result = ackley_function(x[0], x[1])
        final_values_1d.append(result)
        print(gradient_descent.get_parameters())
        print(f"final value: {np.round(result, 2)}, punkt: {np.round(x[0], 2)}")

    # 1D Wykres Zbieżności
    visualization_convergence_ackley_function(history_1d)

    # 1D Wykres trajektoria minimalizacji z najlepszym krokiem
    best_index = np.argmin(final_values_1d)
    optimal_step = steps[best_index]
    gradient_descent = GradientDescent(r_min, r_max, optimal_step, max_iter)
    final_pos, history_f, path_x, path_y = gradient_descent.solve_with_trajectory(ackley_function, x0, y0)
    visualize_best_step_2d(final_pos, history_f, path_x, optimal_step)

    # TODO zbadać wpływ punktu początkowego
    x0 = 15.0
    y0 = 15.0
    history_2d = {}
    final_values_2d = []

    # 2D Wykres funkcji
    # visualization_ackley_function_3d(r_min, r_max)

    # print("\nZbieranie danych dla two_dimensional_ackley_function...")
    # for step in steps:
    #     gradient_descent = GradientDescent(r_min, r_max, step, max_iter)
    #     x, history = gradient_descent.solve(ackley_function_2d, x0, y0)
    #     history_2d[step] = history
    #     result = ackley_function_2d(x[0], x[1])
    #     final_values_2d.append(result)
    #
    #     print(f"2D - step: {step}, final value: {np.round(result, 2)}")

    # 2D Wykres Zbieżności
    # visualization_convergence_ackley_function_2d(history_2d)

    # 2D Wykres Skuteczności Kroku
    # plt.figure(figsize=(12, 7))
    # plt.plot(steps, final_values_2d, marker='o', linestyle='-')
    #
    # plt.title('Końcowa wartość funkcji Ackleya (2D) vs. Wielkość Kroku')
    # plt.xlabel('Wielkość Kroku (Step Size)')
    # plt.ylabel('Osiągnięta Wartość Funkcji Celu (Min)')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()

    # Tworzenie wykresu 3D
    # best_index = np.argmin(final_values_2d)
    # optimal_step = steps[best_index]
    # gradient_descent = GradientDescent(r_min, r_max, optimal_step, max_iter)
    # final_pos, history_f, path_x, path_y = gradient_descent.solve_with_trajectory(ackley_function_2d, x0, y0)
    #
    # visualize_best_step_3d(final_pos, history_f, path_x, path_y, optimal_step)

if __name__ == "__main__":
    main()
