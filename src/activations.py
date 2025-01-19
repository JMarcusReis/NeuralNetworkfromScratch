import numpy as np
from layer import Layer

def sigmoid(x):
    """
    Returns the sigmoid function given by
    sigmoid(x) = 1/(1 + e^(-x))
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    Returns the derivative of the sigmoid function
    """
    return np.exp(-x)/((1 + np.exp(-x)) ** 2)

def tanh(x):
    """
    Returns the hiperbolycal tanget of x
    """
    return np.tanh(x)

def tanh_prime(x):
    """
    Returns the derivative of the hyperbolic tangent
    """
    return 1 - tanh(x) ** 2

def relu(x):
    """
    Returns the value of the RELU function.
    """
    return np.maximum(x, 0)

def relu_prime(x):
    """
    Returns the derivative of the RELU function
    """
    return np.array(x >= 0).astype('int')
