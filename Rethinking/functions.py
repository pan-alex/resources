# Plot the scatter plot / line
import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


def plot_regression_line(x, y, mu, hdpi=0.1, xlab='', ylab=''):
    '''
    x: The predictor variable
    y: The  response variable
    mu: The mu value from the PyMC3 model trace. (eg., trace['mu'])
    hdpi: The alpha value for the HDPI. 0.1 corresponds to the 90% interval
    
    Plots a scatter plot of the data and then the regression line with the HDPI interval shaded.
    '''
    mu_hpd = pm.hpd(mu, alpha = hdpi)
    
    plt.scatter(x, y, alpha = 0.5)
    plt.plot(x, mu.mean(0), 'C2')    # MAP line (column-wise mean of mu)
    
    # HPDI fill-in
    index = np.argsort(x)
    plt.fill_between(x[index], 
                     mu_hpd[:, 0][index],
                     mu_hpd[:, 1][index],
                     color='C2',
                     alpha=0.25)
    plt.xlabel(str(xlab))
    plt.ylabel(str(ylab))
    
    
# Counterfactual

def plot_counterfactual(data, trace, variables, parameters, intercept='a', hpdi=0.10, xlab='', ylab=''):
    """
    x = predictor of interest
    intercept = The string used to denote the intercept (i.e., alpha, or beta_0). Defaults to 'a'
    variables and parameters are lists that must be in the same order (corresponding data column + coefficient). 
    The first value in each list should be the predictor of interest. Do not include the intercept term.
    """
    # Calculate value of x term (predictor of interest)
    x = data[variables[0]]
    x_coef = parameters[0]
    x_space = np.linspace(x.min(), x.max(), 50)
    x_value = trace[x_coef] * x_space[:,None]
    
    # Calculate value of other variables, holding them to the mean value.
    controls = []
    for item in variables[1:]: controls.append(data[item].mean())
    
    control_coefficients = []
    for item in parameters[1:]: control_coefficients.append(trace[item])
        
    control_values = np.multiply(controls, control_coefficients)
    
    # Calculate the predicted mean.
    mu_predicted = trace[intercept] + x_value + control_values
    
    mu_hpd = pm.hpd(mu_predicted.T, alpha=hpdi)

    plt.plot(x_space, mu_predicted.mean(1), 'k')
    plt.plot(x_space, mu_hpd[:,0], 'k--')
    plt.plot(x_space, mu_hpd[:,1], 'k--')

    plt.xlabel(xlab);
    plt.ylabel(ylab);