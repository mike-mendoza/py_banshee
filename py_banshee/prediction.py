# -*- coding: utf-8 -*-
"""
Py_BANSHEE
Authors: Paul Koot, Miguel Angel Mendoza-Lugo, Dominik Paprotny,
         Elisa Ragno, Oswaldo Morales-Nápoles, Daniël Worm

E-mail:  m.a.mendozalugo@tudelft.nl, paulkoot6@gmail.com & O.MoralesNapoles@tudelft.nl

"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_banshee.rankcorr import ranktopearson, list_dif
from scipy.stats import norm
from scipy.interpolate import interp1d


def inference(Nodes, Values, R, DATA, Output='full',
              SampleSize=1000, Interp='next',
              empirical_data=True, distributions=[], parameters=[]):
    """
    inference makes inference using a non-parametric Bayesian Network
    
    inference conditionalizes a non-parametric
    Bayesian Network (NPBN) defined by the conditional rank correlation 
    matrix R and quantified by dataframe DATA. The NPBN is conditiona-
    lized for each node defined in list Nodes. Inference is made for
    each column not included in list Nodes and for each row in dataframe 
    Values.
    Parameters
    ----------
    Nodes : list
        A list defining nodes to be conditionalized. The
        values of the list define the variables in the
        same order as in dataframe R. At least one node has to
        be left out from Nodes in order to make inferences
        at least for this one node. For example, if R is a
        5-by-5 numpy array, Nodes = [0, 2, 4] will conditionalize
        the BN using the first, third and fifth node and
        make inference of the second and fourth node.
    Values : pandas.core.frame.DataFrame
        A dataframe containing data on which the inference
        will be based upon. Data for each node need to be
        located in columns in the same order as specified 
        in Nodes. The number of columns need to 
        equal the number of nodes specified in Nodes.
    R : np.ndarray
        A matrix generated using bn_rankcorr function.
    DATA : pandas.core.frame.DataFrame
        A matrix containing data for quantifying the NPBN.
        Data for each node need to be located in columns in
        the same order as specified in R. The number of 
        columns need to be equal the number of nodes 
        specified in R. 
    Output : string
        A string setting the type of output of the
        function:
        'full'   provides a cell array with the
                 conditional empirical distributions
                 (default).
        'mean'   provides a matrix with the mean of the
                 conditional empirical distributions.
        'median' provides a matrix with the median of the
                 conditional empirical distributions.
    SampleSize : int
        Number of samples drawn when conditionalizing the
        NPBN. 1000 is the default.
    Interp : string
        A string with the name of the interpolation method:
        'linear', 'nearest', 'next', 'previous', 'spline',
        'pchip', 'cubic', 'v5cubic' or 'makima'. 'Next' is
        the default.
    empirical_data :  bool,The default is True.
        True = DATA is a pd.DataFrame with empirical observations
        if True distributions and parameters should be empty i.e.
        distributions=[], parameters=[]
        False = Nodes are parametric distributions, no empirical observations available
    distributions : list
        A list of strings with the names of the parametric distributions per node
        example: ['norm','genextreme','norm']	(Continuous Statistical functions of scipy.stats)
    parameters : 
        A list of lists float with the corresponding parameters of the parametric 
        distributions per node, example: [100,23],[-0.15,130,50],[500,100]]

    
    Returns
    -------
    F : numpy.ndarray
        By default, provides an array with the 
        conditional empirical distributions for each row in
        Values and each node not specified in Nodes.
    """

    # Defining nodes to be predicted and their number
    remaining_nodes = list_dif(list(range(R.shape[0])), Nodes)
    nr_remaining_nodes = len(remaining_nodes)

    if type(Values) == list:
        Values = np.array(Values).reshape(1, -1)

    n_values = np.shape(Values)[0]

    if Output == 'full':
        F = np.zeros((n_values, nr_remaining_nodes, SampleSize))
    else:
        F = np.zeros((n_values, nr_remaining_nodes))

    # Adding additional edge cases for the marginal empirical distributions
    # in order to avoid NaN values when conditionalizing
    if empirical_data:
        m1 = (DATA.min() - 0.1).to_frame().T
        m2 = (DATA.max() + 0.1).to_frame().T

        DATA = pd.concat([m2, DATA, m1]).reset_index(drop=True)
    else:
        DATA = []

    # # Obtaining the number of nodes (according to the correlation matrix R)
    n_nodes = np.shape(R)[0]

    if empirical_data:
        count = 0
        x = [None] * n_nodes
        f = [None] * n_nodes
        for node in DATA:
            f[count], x[count] = ecdf(DATA[node])
            count += 1
    else:
        dists, params = make_dist(distributions, parameters)

    if not empirical_data:
        if len(distributions) != n_nodes:
            raise Exception('Please check the distributions and parameters')

    # Transforming Spearman's rank correlation into Pearson's correlation 
    # (auxiliary function 1 in bn_rankcorr.py)
    rpearson = ranktopearson(R)

    # Loop for inference for each row in VALUES
    for j in range(n_values):
        # Obtaining the conditional inverse normal distributions at each node
        NormalCond = np.zeros(len(Nodes))  # preallocation

        if empirical_data:
            for i in range(len(Nodes)):
                # Create index i_nodes, who points to the correct column within
                # lists x and f. Necessary when columns to predict are not the last
                # columns in DATA.
                i_nodes = Nodes[i]
                x_int = [x[i_nodes][0] - (x[i_nodes][1] - x[i_nodes][0])] + x[i_nodes][1:]
                y_int = [0] + f[i_nodes][1:]
                f_int = interp1d(x_int, y_int)
                NormalCond[i] = norm.ppf(f_int(Values[j, i]))
        else:
            for i in range(len(Nodes)):
                NormalCond[i] = norm.ppf(dists[Nodes[i]].cdf(Values[j, i], *params[Nodes[i]]))

        # Calculating the parameters of the conditional normal distribution 
        # (auxiliary function 2)
        M_c, S_c = ConditionalNormal(np.zeros(n_nodes), rpearson, Nodes, NormalCond)

        # Sometimes S_c just fails the symmetry test because S_c' differs 
        # slightly from S_c due to numerical errors. Therefore, S_c is 
        # symmetrized in the next step:
        S_c_symm = (S_c + S_c.transpose()) / 2

        # Sampling the conditional normal distribution
        norm_samples = np.random.multivariate_normal(M_c, S_c_symm, SampleSize)

        # Extracting values of the empirical marginal distributions using the
        # probability density function of the conditional normal distribution.
        # The calculation uses auxiliary function 3
        F0 = [None] * nr_remaining_nodes  # preallocation

        for i in range(nr_remaining_nodes):
            if empirical_data:
                F0[i] = inv_empirical(norm.cdf(norm_samples[:, i]),
                                      [f[remaining_nodes[i]], x[remaining_nodes[i]]],
                                      Interp)
            else:
                F0[i] = dists[remaining_nodes[i]].ppf(norm.cdf(norm_samples[:, i]), *params[remaining_nodes[i]])

        if Output == 'full':
            for i in range(nr_remaining_nodes):
                F[j, i, :] = F0[i]
        elif Output == 'mean':
            for i in range(nr_remaining_nodes):
                F[j, i] = np.mean(F0[i])

        if n_values > 100:
            txt = 'Making inference. Progress: '
            prog = np.floor(j / n_values * 100)
            print('%s %d%%' % (txt, prog))

    return F


# -------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# -------------------------------------------------------------------------
#
# -------------------------------------------------------------------------
# 1 - Calculating the marginal Empirical Cumulative Distribution Function
# -------------------------------------------------------------------------
def ecdf(column):
    """
    ecdf produces the marginal Empirical Cumulative Distribution Function
    
    This function corresponds with MATLABs ecdf function
    Parameters
    ----------
    column : pandas.core.series.Series
        column from a DataFrame to produce the ECDF from
    Returns
    -------
    f : list
        ECDF, evaluated in the points x
    x : list
        Points in which the ECDF is evaluated
    """
    sq = column.value_counts()
    a = sq.sort_index().cumsum() * 1. / len(column)
    f = a.tolist()
    x = a.index.values.tolist()
    # Add a starting point to ecdf to make ecdf start at f=0 
    f = [0] + f
    x = [x[0]] + x
    return f, x


# -------------------------------------------------------------------------
# 2 - Calculating the parameters of the conditional normal distribution
# -------------------------------------------------------------------------
def ConditionalNormal(M, S, idxC, valC):
    """
    CondionalNormal calculates the parameters of the conditional 
    normal distribution
    
    Parameters
    ----------
    M : numpy.ndarray
        Mean vector of the multivariate normal
    S : numpy.ndarray
        Covariance matrix of the multivariate normal
    idxC : list
        Index/Indices of the conditioning nodes
    valC : numpy.ndarray
        Values of the conditioning nodes
        
    Returns
    -------
    Mc : numpy.ndarray
        Mean vector of the conditional multivariate normal on valC
    Sc : numpy.ndarray
        Covariance matrix of the conditional multivariate normal valC
    """

    D = len(M)  # Dimension of the multivariate normal
    idxNC = list_dif(range(D), idxC)  # Index of all the remaining variables

    # Calculation of the conditional normal distribution:
    M1 = M[idxNC]
    S11 = S[np.ix_(idxNC, idxNC)]
    X2 = valC
    M2 = M[idxC]
    S22 = S[np.ix_(idxC, idxC)]
    S12 = S[np.ix_(idxNC, idxC)]
    S21 = S[np.ix_(idxC, idxNC)]
    S22_inv = np.linalg.inv(S22)

    Sc = S11 - S12 @ S22_inv @ S21
    Mc = M1 + S12 @ S22_inv @ (X2 - M2)
    return Mc, Sc


# -------------------------------------------------------------------------
# 3 - Calculating the inverse of the conditional empirical distribution
# -------------------------------------------------------------------------
def inv_empirical(yi, empcdf, way):
    """
    inv_empirical calculates the inverse of the conditional empirical 
    distribution
    
    Parameters
    ----------
    yi : numpy.ndarray
        The samples of the conditional normal distribution
    empcdf : list
        The empirical marginal distribution of a given node (both f and x)
    way : string
        Interpolation method in interp1 function ('next' is default)
        

    Returns
    -------
    xi : 
        Inverse of the conditional empirical distribution
        
    
    """

    fe = empcdf[0]  # cumulative probability density
    xe = empcdf[1]  # corresponding empirical values

    func_i = interp1d(fe, xe, kind=way)  # interpolation

    xi = func_i(yi)
    return xi


# -------------------------------------------------------------------------
# 4 - Parametric distributions
# -------------------------------------------------------------------------


def make_dist(distributions, parameters):
    '''
    make_dist convert the input distributions to a Continuous distributions scipy.stats object 

    Parameters
    ----------
    distributions : list
        A list of the names of the distributions fro each node
    parameters : TYPE list
        A list of lists of  the corresponding parameters of the distributions

    Returns
    -------
    dists : list
       A list of scipy distributions objects
    params : list
       A list of list with the parameters of the distributions 

    '''
    # Get all continuous distributions from scipy.stats   
    dist_all = [getattr(scipy.stats, d) for d in dir(scipy.stats) if isinstance(getattr(scipy.stats, d), scipy.stats.rv_continuous)]
    # Create a dictionary for fast lookup by distribution name
    dist_dict = {dist.name: dist for dist in dist_all}
    # Get the distribution objects to test
    dists = [dist_dict[dist_name] for dist_name in distributions if dist_name in dist_dict]
    # Create parameter tuples
    params = [tuple(param) for param in parameters]

    return dists, params


# -------------------------------------------------------------------------
# 5 - un-conditional and conditional marginal histograms
# -------------------------------------------------------------------------

def conditional_margins_hist(F, DATA, names, condition_nodes, empirical_data=True, distributions=[], parameters=[]):
    """
    conditional_margins_hist shows the histogram comparison plots of th conditional and unconditional samples

    Parameters
    ----------
    F : numpy.ndarray
        By default, provides an array with the
        conditional empirical distributions for each row in
        Values and each node not specified in Nodes.
    DATA : pandas.core.frame.DataFrame
        A matrix containing data for quantifying the NPBN.
        Data for each node need to be located in columns in
        the same order as specified in R. The number of
        columns need to be equal the number of nodes
        specified in R.
    names : list
        A list of str with the names of the nodes
    condition_nodes : list
        A list defining nodes to be conditionalized. The
        values of the list define the variables in the
        same order as in dataframe R.
    empirical_data : bool
        True = DATA is a pd.DataFrame with empirical observations
        if True distributions and parameters should be empty i.e.
        distributions=[], parameters=[]
        False = Nodes are parametric distributions, no empirical observations available
    distributions : list
        A list of the names of the distributions fro each node
        The default is [].
    parameters :  list
        A lsit of scipy distributions objects
        The default is [].

    Raises
    ------
    Exception 'Check if argument Output in inference is equal to full'
       F should provide a cell array with the conditional empirical distributions
       options  mean, median in inference function will rise the exception

    Returns
    -------
    None. Shows the plots

    """
    remaining_nodes = list_dif(list(range(len(names))), condition_nodes)
    nr_remaining_nodes = len(remaining_nodes)

    if empirical_data:
        try:
            F_uncond = DATA.iloc[:, remaining_nodes].to_numpy()
            F_cond = np.array(F[0]).transpose()
            for i in range(nr_remaining_nodes):
                F_cond[:, i]
                plt.figure(figsize=(7,4))
                plt.hist(F_uncond[:, i], bins=20, edgecolor="silver", color='silver',
                         label=['un-conditionalized\n mean: ' + str(round(np.mean(F_uncond[:, i]), 1))])
                plt.hist(F_cond[:, i], bins=20, edgecolor="cornflowerblue", color='cornflowerblue',
                         label=['conditionalized\n mean: ' + str(round(np.mean(F_cond[:, i]), 1))])
                plt.legend(fontsize=18)
                plt.ylabel("Count", fontsize=18 )
                plt.xlabel("x", fontsize=18)
                plt.title(names[remaining_nodes[i]], fontsize=20, fontweight=550)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.tight_layout()
                plt.savefig(names[remaining_nodes[i]] + '_uncond_cond.pdf')
                plt.show()
                
        
        except:
            raise Exception('Check if argument Output in inference is equal to full')
    else:
        try:
            # Random unconditional samples of the parametric nodes
            dists, params = make_dist(distributions, parameters)
            random_values = [dists[i].rvs(*params[i], F[0].shape[1]) for i in range(len(dists))]
            random_values = pd.DataFrame(np.array(random_values).transpose(), columns=names)

            F_uncond = random_values.iloc[:, remaining_nodes].to_numpy()
            F_cond = np.array(F[0]).transpose()
            for i in range(nr_remaining_nodes):
                F_cond[:, i]
                plt.figure(figsize=(7, 4))
                plt.hist(F_uncond[:, i], bins=20, edgecolor='silver', color='silver',
                         label=['un-conditionalized\n mean: ' + str(round(np.mean(F_uncond[:, i]), 1))])
                plt.hist(F_cond[:, i], bins=20, edgecolor="cornflowerblue", color='cornflowerblue',
                         label=['conditionalized\n mean: ' + str(round(np.mean(F_cond[:, i]), 1))])
                plt.legend(fontsize=18)
                plt.ylabel("Count", fontsize=18 )
                plt.xlabel("x", fontsize=18)
                plt.title(names[remaining_nodes[i]], fontsize=20, fontweight=550)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.tight_layout()
                plt.savefig(names[remaining_nodes[i]] + '_uncond_cond.pdf')
                plt.show()
				
        except:
            raise Exception('Check if argument Output in inference is equal to full')
    return None
