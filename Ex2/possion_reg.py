# Author: Johanna Rissbacher
# Date: 12/09/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln


# define functions for possion regression
def likelihood_func(beta):
    return np.sum(-np.exp(beta[0]+beta[1]*x) + y*(beta[0]+beta[1]*x) - gammaln(y+1))

def neg_likelihood_func(beta):
    return -likelihood_func(beta)

def poisson_reg(x, beta):
    return np.exp(beta[0] + beta[1]*x)


# load bird count data
data = pd.read_csv('bird_count.csv')
x = data['yr'].values
y = data['count'].values

# initial guess
beta_0 = np.array([np.log(np.mean(y)), 0])

# Maximum liklihood estimation
res = minimize(neg_likelihood_func, beta_0)

# predictions according to the possion regression
pred = poisson_reg(x, np.array([res.x[0], res.x[1]]))

# fix array for plotting
order = np.argsort(x)
x_sorted = x[order]
y_pred_sorted = pred[order]

# plot the results
plt.plot(data['yr'], data['count'], 'o', label='og data')
plt.plot(data['yr'], pred, 'o', label='poisson regression', color='orange')
plt.plot(x_sorted, y_pred_sorted, color='orange')
plt.xlabel('Year')
plt.ylabel('Bird Count')
plt.legend()
plt.title('Poisson Regression of Bird Count Data')
plt.show()


# Generate 3 samples of data
# Predictions for the years 2000, 2004, 2006
pred_years = np.array([2000, 2004, 2006])
pred_counts = poisson_reg(pred_years, np.array([res.x[0], res.x[1]]))

samples_pred = pd.DataFrame({'Year': pred_years, 'Predicted Count': np.round(pred_counts, 2)})
samples_pred.to_csv('poisson_regression_predictions.csv', index=False)
