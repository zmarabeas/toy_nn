import numpy as np

def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        return (1 / (1 + np.exp(-x)))

def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x)**2
    else:
        return np.tanh(x)


# def relu(x, derivative=False):
#     if derivative:
#         x[x <= 0] = 0
#         x[x > 0] = 1
#         return x
#     else:
# 	    return np.maximum(x, 0)


def relu(x, derivative=False):
    if derivative:
        return (x > 0) * 1
    else:
        return x * (x > 0)


def leaky_relu(x, derivative=False, alpha=.01):
    if derivative:
        x[x > 0] = 1
        x[x <= 0] = alpha
        return x
    else:
        x[x<=0] *= alpha
        return x
