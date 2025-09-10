import math
import numpy as np
from scipy.optimize import minimize

def LOESS(x, y, k, x_0):
    """
    Args:
        x (_type_): vector of obs of the predictor
        y (_type_): vector of obs of the response variable
        k (_type_): number of neighboring points
        x_0 (_type_): vector of values for which a prediction is going to be made
    
    Returns:
        pred (_type_): vector of predicted values
        se (_type_): vector of standard deviations of the expected value of each predicted value
    """

    distances = np.zeros(len(x))
    pred = np.zeros(len(x_0))
    se = np.zeros(len(x_0))
    for i in range(len(x_0)):
        # select the k nearest points to x_0[i]
        #distances[i] = abs(x - x_0[i])
        # for j in range(len(x)):
        #     distances[j] = abs(x[j], x_0[i])
        distances[:] = np.abs(x - x_0[i])
        idx = np.argsort(distances)[:k]
        x_k = x[idx]
        y_k = y[idx]

        # compute the weights
        # normalize distances
        distances_k = distances[idx]
        distances_k_normalized = (distances_k - np.min(distances_k)) / (np.max(distances_k) - np.min(distances_k))
        weights = (1 - abs(distances_k_normalized)**3)**3

        # fit a weighted OLS
        beta_0 = np.array([940, 0.1]) # initial guess
        def Q(beta):
            return np.sum(weights*(y_k - beta[0] - beta[1] * x_k)**2)
        res = minimize(Q, beta_0)
        pred[i] = res.x[0] + res.x[1] * x_0[i]

        # compute the standard deviations of the expected value of each predicted value
        pred_k = res.x[0] + res.x[1] * x_k # prediction for k nearest points

        RSS = np.sum((y_k - pred_k)**2)
        var = RSS / (k-2)
        n = len(y)
        standard_deviation = np.sqrt(var * (1/n + (x_0[i] - np.mean(x_k))**2 / np.sum((x - np.mean(x_k))**2)))
        se[i] = standard_deviation
  

    return pred, se