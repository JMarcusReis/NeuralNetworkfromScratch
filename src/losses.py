import numpy as np

def mse(y_true, y_pred):
    """
    Returns te Mean Square Error given the formula
    MSE = ((y1 -^y1)**2 + (y2 -^y2)**2 + ... (yn -^yn)**2)/n
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """
    Returns the derivative of MSE
    """
    return 2 * (y_true - y_prime)/y_true.size
