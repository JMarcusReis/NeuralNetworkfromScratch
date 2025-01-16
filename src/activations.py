import numpy as np
from layer import Layer

def sigmoid(x):
    """
    Returns the sigmoid function given by
    sigmoid(x) = 1/(1 + e^(-x))
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """"""
    return np.exp(-x)/((1 + np.exp(-x)) ** 2)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - tanh(x) ** 2

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')
