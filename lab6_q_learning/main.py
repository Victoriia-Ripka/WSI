"""
Author: Viktoriia Nowotka
"""
from lab6_q_learning.q_learn import QLearning


def main():
    t_max = 1
    e_max = 1
    U = [0, 1, 2]
    n_dirs = 4

    # Wykonaj przy tym eksperymenty dla różnych zestawów parametrów, uwzględniając:
    eps = 0.01              # parametr eksploracji ϵ
    b = 0.1                 # szybkość uczenia β
    y = 0.9                 # współczynnik dyskontowania γ

    ql = QLearning(U, n_dirs, e_max, t_max)
    ql.train(b, y, eps)
    ql.close()

if __name__ == "__main__":
    main()



