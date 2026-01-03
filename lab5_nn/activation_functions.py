import numpy as np

def relu(x, derive=False):
    if not derive:
        return np.maximum(0, x)
    return np.where(x >= 0, 1, 0)


def sigmoid(x, derive=False):
    clipped_x = np.clip(x, -500, 500)
    if not derive:
        return 1 / (1 + np.exp(-clipped_x))
    return x * (1 - x)


def tanh(x, derive=False):
    if not derive:
        return np.tanh(x)
    return 1 - np.square(np.tanh(x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
