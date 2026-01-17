"""
Author: Viktoriia Nowotka
"""
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

# Q-learning z epizodami oraz ϵ-zachłanną strategią wyboru akcji
# * Dostosuj rozwiązanie, aby działało również dla losowych pozycji agenta i celu.
# Terminations:
# The agent reaches the goal.
# Timeout (see max_steps).

class QLearning:
    def __init__(self, actions, n_dirs, e_max, t_max):
        self.e_max = e_max
        self.t_max = t_max

        self.env = gym.make("MiniGrid-FourRooms-v0", max_steps=10, goal_pos=(2, 2), agent_pos=(7, 7), render_mode="human")
        self.state_to_id = self.__build_state_mapping()

        self.U = actions            # actions: left, right, forward
        self.n_dirs = n_dirs        # 4 directions
        self.__init_states()        # available states: (x, y, dir)

    # TODO REFACTOR
    def __init_states(self):
        self.X = []

        base_env = self.env.unwrapped
        for x in range(base_env.width):
            for y in range(base_env.height):
                cell = base_env.grid.get(x, y)
                if cell is None or cell.can_overlap():
                    self.X.append((x, y))

    def __build_state_mapping(self):
        base_env = self.env.unwrapped
        state_to_id = {}
        idx = 0

        for x in range(base_env.width):
            for y in range(base_env.height):
                cell = base_env.grid.get(x, y)
                if cell is None or cell.can_overlap():
                    for d in range(4):
                        state_to_id[(x, y, d)] = idx
                        idx += 1

        return state_to_id

    def __define_x(self):
        base_env = self.env.unwrapped
        x, y = base_env.agent_pos
        d = base_env.agent_dir

        return self.state_to_id[(x, y, d)]

    def choose_action(self, x, Q, eps):
        if np.random.random() < eps:
            return np.random.choice(self.U)     # eksploracja
        else:
            return np.argmax(Q[x])              # eksploatacja

    def train(self, b, y, eps=0.1):
        e = 0
        Q = np.zeros((len(self.X) * self.n_dirs, len(self.U)))

        while e < self.e_max:

            t = 0
            obs, info = self.env.reset(seed=42)
            x = self.__define_x()

            while t < self.t_max:
                act = self.choose_action(x, Q, eps)
                obs, rew, terminated, truncated, info = self.env.step(act)
                x = self.__define_x()

                delta = rew + y * max(Q[x, act]) - Q[x, act]
                Q[x, act] += b * delta
                t += 1

                if terminated or truncated:
                    x, info = self.env.reset()
            e += 1
        self.env.reset()

    def run(self):
        pass

    def close(self):
        self.env.close()