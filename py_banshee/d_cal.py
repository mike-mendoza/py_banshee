# -*- coding: utf-8 -*-
"""
Py_BANSHEE
Authors: Paul Koot, Miguel Angel Mendoza-Lugo, Dominik Paprotny,
         Elisa Ragno, Oswaldo Morales-Nápoles, Daniël Worm

E-mail:  m.a.mendozalugo@tudelft.nl, paulkoot6@gmail.com & O.MoralesNapoles@tudelft.nl

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from py_banshee.rankcorr import NormalTransform, pearsontorank, ranktopearson
from py_banshee.prediction import ecdf


def gaussian_distance(R, DATA, SampleSize_1=1000,
                      SampleSize_2=1000, M=1000, Plot=False, Type='H', fig_name=''):
    """
    gaussian_distance measures the distance between Gaussian densities
    
    gaussian_distance calculates the distance between the 
    empirical (ERC) and Bayesian Network's (BNRC) rank correlation matrices 
    and the empirical normal rank correlation matrix (NRC) based on DATA. 
    The resulting d-calibration score D and quantile ranges B of 
    determinants of the empirical normal and Bayesian Network's distributions 
    can be used to verify to what extent the assumption of joint normal copula 
    is valid.

    Parameters
    ----------
    R : np.ndarray
        A rank correlation matrix generated using bn_rankcorr function
    DATA : pandas.core.frame.DataFrame
        A matrix containing data for quantifying the NPBN.
        Data for each node need to be located in columns in
        the same order as specified in R. The number of 
        columns need to be equal to the number of nodes 
        specified in R.
    SampleSize_1 : int
        The number of samples to be drawn in
        the resampling of the distributions in the test
        d-Cal(ERC,NRC). 1000 is the default.
    SampleSize_2 : int
        The number of samples to be drawn in
        the resampling of the distributions in the test   
        d-Cal(NRC,BNRC). 1000 is the default.   
    M : int
        Number of iterations of calculating the d-calibration
        scores to compute the confidence interval
        of the determinant of the sampled random distribution.
        1000 is the default.
    Plot : bool
        A plot highlighting the d-calibration scores can be
        displayed. The options are:
        False - do not create a plot (default)
        True - create a plot.
    Type : string
        Type of measure used to calculate the distance.
        Available methods are:
        'H'       Hellinger distance (default)
        'KL'      Symmetric Kullback–Leibler divergence
        'B'       Bhattacharyya distance
        'G'       G distance (Abou Moustafa et al. 2010)
    fig_name : string
        Name extension of the .png file with the d-calibration scores that
        is created: gaussian_distance_'fig_name'.png. 
        The file is saved in the working directory.
    
    Returns
    -------
    D_ERC : numpy.ndarray
        A numeric value of the d-calibration score for the 
        empirical rank correlation matrix of DATA
    D_BNRC : numpy.ndarray
        A numeric value of the d-calibration score for the 
        Bayesian Network rank correlation matrix R.
    B_ERC : numpy.ndarray
        Quantile range (5th and 95th percentile) of the
        distribution of the determinant of the empirical
        distribution of DATA transformed to standard normal.
    B_BNRC : numpy.ndarray
        Quantile range (5th and 95th percentile) of the
        distribution of the determinant of the empirical
        distribution of the Bayesian Network.

        The score is 1 if the matrices are equal and 0 if 
        one matrix contains a pair of variables perfectly 
        correlated, and the other one does not, and the 
        score will be “small” as the matrices differ from 
        each other elementwise.
    """
    DATA = DATA.dropna()  # remove NaN values

    # Reading the number of variables
    Nvar = DATA.shape[1]

    # Computing empirical normal rank correlation matrix (NRC)
    [Z, U] = NormalTransform(DATA)  # transforming data to standard normal
    rho = np.corrcoef(Z, rowvar=False)  # calculating Pearson's correlation
    Sigma2 = pearsontorank(rho)  # transforming Pearson's to Spearman's

    # Computing the empirical rank correlation matrix (ERC).
    # The empirical distribution based on the rank (U) was obtained already
    # through auxiliary function 1.
    Sigma1 = np.corrcoef(U, rowvar=False)

    # Bayesian Network's rank correlation matrix (BNRC)
    Rbn = R

    # d-calibration scores (auxiliary function 1)
    D_ERC = np.squeeze(1 - test_distance(Sigma1, Sigma2, Type, Nvar))
    D_BNRC = np.squeeze(1 - test_distance(Rbn, Sigma2, Type, Nvar))

    # Transforming the Bayesian Network's rank correlation matrix to product
    # moment correlation.
    RHO = ranktopearson(Rbn)

    # Preallocate D_NR and D_BN
    D_NR = np.zeros(M)
    D_BN = np.zeros(M)

    # Calculating the range of d-calibration scores for distributions sampled
    # from the empirical normal and Bayesian Network
    for i in range(M):
        # d-calibration for NRC
        # drawing random normal samples from the empirical correlation matrix
        Z1 = np.random.multivariate_normal(np.zeros(Nvar), RHO, SampleSize_1)
        # computing correlation matrix for the sampled normal distribution
        RHO1 = np.corrcoef(Z1, rowvar=False)
        # transforming product moment correlation to rank correlation
        S1 = pearsontorank(RHO1)
        # repeated for another set of samples
        Z2 = np.random.multivariate_normal(np.zeros(Nvar), RHO, SampleSize_1)
        RHO2 = np.corrcoef(Z2, rowvar=False)
        S2 = pearsontorank(RHO2)

        # d-calibration score computed with auxiliary function 2
        D_NR[i] = 1 - test_distance(S1, S2, Type, Nvar)

        # d-calibration for BNRC
        # drawing random normal samples from the BN's correlation matrix
        Z3 = np.random.multivariate_normal(np.zeros(Nvar), RHO, SampleSize_2)
        RHO3 = np.corrcoef(Z3, rowvar=False)
        S3 = pearsontorank(RHO3)
        # repeated for another set of samples
        Z4 = np.random.multivariate_normal(np.zeros(Nvar), RHO, SampleSize_2)
        RHO4 = np.corrcoef(Z4, rowvar=False)
        S4 = pearsontorank(RHO4)
        # d-calibration score computed with auxiliary function 2
        D_BN[i] = 1 - test_distance(S3, S4, Type, Nvar)

    # quantile ranges
    q = [.05, .95]  # defining the quantile range
    B_ERC = np.quantile(D_NR, q)
    B_BNRC = np.quantile(D_BN, q)

    # Generate plot
    if Plot:
        # first subplot with the d-calibration and range for the ERC
        f, x = ecdf(pd.Series(D_NR))

        fig, ax = plt.subplots(figsize=(12, 6), sharex=False, sharey=False, ncols=2, nrows=1)
        ax[0].step(np.array(x), np.array(f), color='k')
        ax[0].plot([D_ERC, D_ERC], [f[1], 1], 'r', linewidth=2)
        ax[0].set_ylim([f[0], f[-1]])
        ax[0].set_xlabel('d-calibration score',fontsize=15)
        ax[0].set_ylabel('Cumulative density function',fontsize=15)
        ax[0].plot(B_ERC, q, 'or', markersize=6)
        ax[0].set_title('d-Cal(ERC,NRC)\n '
                    'in the distribution of\n '
                    'd-Cal(NRC,NRC)\n '+
                    str(SampleSize_1) +' samples in '+ str(M)+'iterations', fontsize=17)
        ax[0].grid(which="both", ls="--", linewidth=.5)
        ax[0].tick_params(axis ='x', labelsize=14, labelrotation=45)
        ax[0].tick_params(axis ='y', labelsize=14)

        
        # second subplot with the d-calibration and range for the BNRC
        f1, x1 = ecdf(pd.Series(D_BN))

        ax[1].step(np.array(x1), np.array(f1), '-k')
        ax[1].plot([D_BNRC, D_BNRC], [f1[1], 1], 'r', linewidth=2)
        ax[1].set_ylim([f1[0], f1[-1]])
        ax[1].set_xlabel('d-calibration score', fontsize=16)
        # ax[0,1].set_ylabel('Cumulative density function')    
        ax[1].plot(B_BNRC, q, 'or', markersize=6)
        ax[1].set_title('d-Cal(NRC,BNRC)\n '
                    'in the distribution of\n '
                    'd-Cal(BNRC,BNRC)\n '+
                    str(SampleSize_2) +' samples in '+ str(M)+'iterations', fontsize=17)
        ax[1].grid(which="both", ls="--", linewidth=.5)
        ax[1].tick_params(axis ='x', labelsize=14, labelrotation=45)
        ax[1].tick_params(axis ='y', labelsize=14)
        plt.tight_layout()
        plt.savefig('gaussian_distance_{}.pdf'.format(fig_name))
        plt.show()

    if D_ERC > B_ERC[0] and D_ERC < B_ERC[1]:
        print(
            'SUCCESS: The d-Cal of the empirical rank correlation matrix (ERC) fall between the confidence intervals of the d-Cal of the normal rank correlation matrix (NRC)\n')
    else:
        print(
            'FAILURE: The d-Cal of the empirical rank correlation matrix (ERC) is out of the confidence intervals of the d-Cal of the normal rank correlation matrix (NRC)\n')

    if D_BNRC > B_BNRC[0] and D_BNRC < B_BNRC[1]:
        print(
            'SUCCESS: The d-Cal of the normal rank correlation matrix (NRC) fall between the confidence intervals of the d-Cal of the BN rank correlation matrix (BNRC)\n')
    else:
        print(
            'FAILURE: The d-Cal of the normal rank correlation matrix (NRC) is out of the confidence intervals of the d-Cal of the BN rank correlation matrix (BNRC)\n')

    return D_ERC, B_ERC, D_BNRC, B_BNRC


# -------------------------------------------------------------------------
# 2 - Compute distance between matrices
# -------------------------------------------------------------------------
def test_distance(Sigma1, Sigma2, Type, Nvar):
    """
    test_distance computes the distance between matrices
    
    Parameters
    ----------
    Sigma1 : numpy.ndarray
        correlation matrix of the first distribution
    Sigma2 : numpy.ndarray
        correlation matrix of the second distribution
    Type : string
        Type of measure used to calculate the distance.
        Available methods are:
        'H'       Hellinger distance (default)
        'KL'      Symmetric Kullback–Leibler divergence
        'B'       Bhattacharyya distance
        'G'       G distance (Abou Moustafa et al. 2010) 
    Nvar : int
        the number of variables in DATA
    
    Returns
    -------
    D : numpy.ndarray
        d-calibration score, the distance between matrices
   
    """
    # Mean vector of the first distribution
    m1 = np.zeros((Nvar, 1))

    # Mean vector of the second distribution
    m2 = np.zeros((Nvar, 1))

    # Distance calculation
    if Type == 'H':
        # Hellinger distance
        # elements of the distance equation
        a = (np.linalg.det(Sigma1) ** (1 / 4) * np.linalg.det(Sigma2) ** (1 / 4)) / \
            np.linalg.det((1 / 2) * Sigma1 + (1 / 2) * Sigma2) ** (1 / 2)
        b = np.exp(-(1 / 8) * (m1 - m2).reshape(1, -1) @ np.linalg.inv((1 / 2) * Sigma1 + (1 / 2) * Sigma2) @ (m1 - m2))
        # equation proper
        D = (1 - (a * b)) ** (1 / 2)
    elif Type == 'KL':
        # Symmetric Kullback–Leibler divergence
        D = (1 / 2) * (m1 - m2).reshape(1, -1) @ (np.linalg.inv(Sigma1) + np.linalg.inv(Sigma2)) @ (m1 - m2) + \
            (1 / 2) * np.trace(np.linalg.inv(Sigma1) * Sigma2 + \
                               Sigma1 * np.linalg.inv(Sigma2) - 2 * np.identity(Sigma1.shape[0]))
    elif Type == 'B':
        # Bhattacharyya distance
        D = (1 / 8) * (m1 - m2).reshape(1, -1) @ np.linalg.inv((1 / 2) * Sigma1 + (1 / 2) * Sigma2) @ (m1 - m2) + \
            (1 / 2) * np.log((np.linalg.det((1 / 2) * Sigma1 + (1 / 2) * Sigma2)) / \
                             ((np.linalg.det(Sigma1)) ** (1 / 2) * (np.linalg.det(Sigma2)) ** (1 / 2)))
    elif Type == 'G':
        # G distance (Abou Moustafa et al. 2010)
        L = eigh(Sigma1, Sigma2, eigvals_only=True)
        D = (m1 - m2).reshape(1, -1) @ np.linalg.inv((1 / 2) * Sigma1 + (1 / 2) * Sigma2) @ (m1 - m2) + \
            (sum((np.square(np.log(L))))) ** (1 / 2)
    else:
        # displating error if no valid distance type was inserted
        print('Error: Type not recognized')

    return D
