import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln


# define functions for possion regression
def likelihood_func(beta, x, y):
    return np.sum(-np.exp(beta[0]+beta[1]*x) + y*(beta[0]+beta[1]*x) - gammaln(y+1))

def neg_likelihood_func(beta, *args):
    return -likelihood_func(beta, *args)

def poisson_reg(x, beta):
    return np.exp(beta[0] + beta[1]*x)