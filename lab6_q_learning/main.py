"""
Author: Viktoriia Nowotka
"""
from lab6_q_learning.q_learn import QLearning


def main():
    U = [0, 1, 2]
    n_dirs = 4

    e_max = 15
    max_steps = 100

    # Wykonaj przy tym eksperymenty dla różnych zestawów parametrów, uwzględniając:
    b = 0.25                 # szybkość uczenia β
    y = 0.95                 # współczynnik dyskontowania γ
    eps = 0.25               # parametr eksploracji ϵ

    ql = QLearning(U, n_dirs, e_max, max_steps)
    ql.train(b, y, eps)
    n_steps = ql.run()
    ql.close()

# współczynnik dyskontowania γ = 1 - 1/H, gdzie H = horyzont kroków.
# Przy H = 10 => γ = 0.9
# Przy H = 20 => γ = 0.95
# Przy H = 40 => γ = 0.975
# Przy H = 100 => γ = 0.99

#  parametr eksploracji ϵ

if __name__ == "__main__":
    main()



