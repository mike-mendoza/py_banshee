# -*- coding: utf-8 -*-
"""
Py_BANSHEE
Authors: Paul Koot, Miguel Angel Mendoza-Lugo, Dominik Paprotny,
         Elisa Ragno, Oswaldo Morales-Nápoles, Daniël Worm

E-mail:  m.a.mendozalugo@tudelft.nl, paulkoot6@gmail.com & O.MoralesNapoles@tudelft.nl
"""

import numpy as np
import scipy as sc
import pandas as pd
from matplotlib import pyplot as plt
from pycopula.copula import ArchimedeanCopula, GaussianCopula
from itertools import combinations


def cvm_statistic(DATA, names, plot=False, fig_name=''):
    """ cvm_statistic calculates the goodness-of-fit for each pair 
    of variables in DATA, using Cramer-von Mises statistic. This statistic
    measures the sum of squared difference between the parametric and 
    empirical copulas and returns the statistics in M.
     
     
    Parameters
    ----------
    DATA : pandas.core.frame.DataFrame
        dataframe containing data to be tested. Each column is one variable
    plot : bool
        A plot highlighting the optimal copula per pair of
        variables can be displayed. The options are:
        False - do not create a plot (default);
        True - create a plot.
    names : list
        list with the names of the variables for the plot
    fig_name : string
        Name extension of the .png file with the statistics that are created: 
        cvm_statistics_'fig_name'.png. The file is saved in the 
        working directory.         
    
    Returns
    -------
    M : numpy.ndarray
        A matrix containing the following columns: 
        Column 1: first variable in the pair
        Column 2: second variable in the pair
        Column 3: Spearman's rank correlation between variables.
        Column 4: Cramer-von Mises statistic for Clayton copula 
        Column 5: Cramer-von Mises statistic for Frank copula 
        Column 6: Cramer-von Mises statistic for Gaussian copula
        Column 7: Cramer-von Mises statistic for Gumbel copula
    """

    # Reading the number of variables
    N = DATA.shape[1]

    # Calculating the number of combinations and an index for looping
    k = 2  # size of a group
    # Number of Combinations
    Nk = int(np.math.factorial(N) / (np.math.factorial(N - k) * np.math.factorial(k)))
    # Vector of indices to calculate the test
    indP = combinations(range(N), k)
    M = np.empty((Nk, 7))  # preallocation
    M[:] = np.nan

    # Calculate an unconditional rank correlation matrix
    DATA2 = DATA.dropna()  # remove NaN values
    Rd = sc.stats.spearmanr(DATA2)[0]  # compute correlation
    Nk_i = 0  # initiate counter to allocate combination in M
    # Calculating the CvM statistics per each pair and copula type
    for i in list(indP):
        # Selecting variables for the pair
        var = DATA.iloc[:, [i[0], i[1]]]
        # Storing the numbers of the variables in the pair
        M[Nk_i, 0] = i[0]
        M[Nk_i, 1] = i[1]
        # Computing rank correlation between variables
        M[Nk_i, 2] = Rd[i[0], i[1]]
        # Computing CvM statistic per pair (auxiliary function 1)
        M[Nk_i, 3] = CVM(var, 'Clayton')
        M[Nk_i, 4] = CVM(var, 'Frank')
        M[Nk_i, 5] = CVM(var, 'Gaussian')
        M[Nk_i, 6] = CVM(var, 'Gumbel')
        Nk_i += 1

    if plot:
        fig, ax = plt.subplots(figsize=(12, 12), sharex=False, sharey=False, ncols=N - 1, nrows=N - 1)
        x = np.arange(4)
        labels = ['Cla', 'Fra', 'Gau', 'Gum']
        colors = ['#1f77b4', '#1f77b4', '#BFE5D9', '#1f77b4']
        count = 0
        for i in range(N - 1):
            for j in range(N - 1):
                if i > j:
                    ax[i, j].axis('off')
                else:
                    ax[i, j].bar(x, M[count, 3:7], color=colors)
                    ax[i, j].set_xticks(x)
                    ax[i, j].set_xticklabels(labels, rotation=90)
                    if i == 0:
                        ax[i, j].set_title(names[int(M[count, 1])])
                    if i == j:
                        ax[i, j].set_ylabel(names[int(M[count, 0])])
                    count += 1

        plt.tight_layout()
        plt.savefig('cvm_statistics_{}.png'.format(fig_name))

    return M


# -------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# -------------------------------------------------------------------------
#
# --------------------------------------------------------------------------
# 1 - Calculating the Cramer-von Mises statistic M
# --------------------------------------------------------------------------
def CVM(OBS, family):
    """ 
    Calculates the Cramer-von Mises statistic M
    
    Parameters
    ----------
    OBS : pandas.core.frame.DataFrame
        nx2 DataFrame of observations
    family : string
        string containing the name of the copula family to be tested
        (1) 'Gaussian', (2)'Frank', (3)'Clayton', (4)'Gumbel'
        
    Returns
    -------
    m : float
        Cramer-von Mises statistic M
    """

    OBS = OBS.dropna()  # remove NaN values

    # Calculate theoretical copula
    Nobs = OBS.shape[0]  # number of observations
    Rnks = OBS.rank(axis=0)  # rank data

    # Uniform Transfer of the observations
    # Note: Clayton and Gumbel copulas cannot handle variables negatively
    # correlated. In case of Clayton or Gumbel family one axis is rotated so 
    # that the correlation becomes positive

    Uobs = pd.DataFrame(np.empty((Nobs, 2)) * np.nan)
    r = sc.stats.pearsonr(OBS.iloc[:, 0], OBS.iloc[:, 1])[0]

    # Checking copula type and correlation for special case of rotating the
    # copula

    if family in ['Clayton', 'Gumbel'] and r < 0:  # special case
        # Rotate one axis
        # Empirical distribution based on the rank
        Uobs.iloc[:, 0] = Rnks.iloc[:, 0] / (Nobs + 1)
        # Rotated empirical distribution based on the rank
        Uobs.iloc[:, 1] = 1 - (Rnks.iloc[:, 1] / (Nobs + 1))

    else:  # other cases
        Uobs = Rnks / (Nobs + 1)
    # Fitting the data to a parametric copula and
    # calculating copula cumulative density function (CDF)

    try:
        C = copulafit(family, Uobs)  # (auxiliary function 3)
    except ValueError as e:
        print(str(e))

        # Calculate the empirical copula (auxiliary function 2)
    EmpC = emp_copula(Uobs)

    # Cramer-von Mises statistic
    m = np.sum((EmpC - C) ** 2)

    return m


# -------------------------------------------------------------------------
# 2 - Calculating the empirical bi-copula
# -------------------------------------------------------------------------
def emp_copula(D):
    """
    Calculates the empirical bi-copula
    
    Parameters
    ----------
    D : pandas.core.frame.DataFrame
        nx2 DataFrame of observations
        
    Returns
    -------
    Y : numpy.ndarray
        Empirical Copula 
    """
    n = D.shape[0]  # number of observations
    BVP = np.zeros(n)  # pre-assigned bivariate probability array

    D = D.to_numpy()

    for i in range(n):
        CD = np.zeros((n, 2))
        CD[:, 0] = np.where(D[:, 0] <= D[i, 0], 1, 0)
        CD[:, 1] = np.where(D[:, 1] <= D[i, 1], 1, 0)
        BVP[i] = sum(CD[:, 0] * CD[:, 1])

    Y = BVP / ((n + 1))

    return Y


# --------------------------------------------------------------------------
# 3 - Copula fit function that uses the fit functions of copulae package
# --------------------------------------------------------------------------
def copulafit(family, Uobs):
    """ 
    Copula fit function that uses the fit functions of copulae package 
    
    Parameters
    ----------
    family : string
        string containing the name of the copula family to be tested
        (1) 'Gaussian', (2)'Frank', (3)'Clayton', (4)'Gumbel'
    Uobs : pandas.core.frame.DataFrame
        nx2 DataFrame of observations
        
    Returns
    -------
    C : numpy.ndarray
        theoretical cdf
    """

    _, dims = Uobs.shape

    if dims != 2:
        raise ValueError('Input Uobs should be nx2')

    if family == 'Gaussian':
        cop = GaussianCopula(dim=dims)  # initializing the copula
        norm_inv = sc.stats.norm.ppf(Uobs.values)
        RhoHat = np.corrcoef(norm_inv, rowvar=False)
        RhoHat = (RhoHat + RhoHat.transpose()) / 2
        cop.R = RhoHat

    elif family == 'Gumbel':
        cop = ArchimedeanCopula(family='gumbel', dim=dims)
    elif family == 'Clayton':
        cop = ArchimedeanCopula(family='clayton', dim=dims)
    elif family == 'Frank':
        cop = ArchimedeanCopula(family='frank', dim=dims)

    else:
        raise ValueError('Copula not included in function copulafit')

    if family == 'Gaussian':
        param = RhoHat
    else:
        parm = cop.fit(Uobs.values, method="cmle")[0]

    C = [cop.cdf(Uobs.values[i, :]) for i in range(len(Uobs.values))]

    return C  # return theoretical cdf
