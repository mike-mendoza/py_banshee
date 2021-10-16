# -*- coding: utf-8 -*-
"""
Py_BANSHEE
Authors: Paul Koot, Miguel Angel Mendoza-Lugo, Dominik Paprotny,
         Elisa Ragno, Oswaldo Morales-Nápoles, Daniël Worm

E-mail:  m.a.mendozalugo@tudelft.nl, paulkoot6@gmail.com & O.MoralesNapoles@tudelft.nl
"""
import argparse
import scipy.stats as st
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt


def bn_rankcorr(parent_cell, data, var_names, is_data=True, plot=False, fig_name=''):
    """
    bn_rankcorr creates a rank correlation matrix R 
    from a defined Bayesian Network (BN) structure parent_cell
    and a data matrix data.
    
    Parameters
    ----------
    parent_cell : list
        A list of lists containing the structure of the BN.
        Each list is a node of the BN and contains a list  
        of the node's parents, defined as a list with 
        reference to particular list(s).
        
        This an example BN with 4 nodes and 6 arcs:
        parent_cell[0] = []         # Node 1 with no parents
        parent_cell[1] = [0, 2]     # Node 2 with two parents 
                                    # (nodes 1 and 3)
        parent_cell[2] = [0]        # Node 3 with one parent
                                    # (node 1)
        parent_cell[3] = [0, 1, 2]  # Node 4 with three parents
                                      (nodes 1, 2 and 3)
        Note: the BN needs to be an acyclic graph!
    data : pandas.core.frame.DataFrame
        By default, a matrix containing data for 
        quantifying the NPBN. Data for each node need to be 
        located in columns in the same order as specified 
        in parent_cell. The number of columns needs to equal 
        the number of nodes specified in 
        parent_cell.
        Optionally, a cell array of rank correlations can
        be used, one conditional correlation per arc, 
        following the same structure as parent_cell.
    is_data : int
        Specifies the input data type:
            0 - cell array DATA contains rank correlations;
            1 - matrix DATA contains actual data.
    plot : bool
        A plot of correlation matrix R can be displayed.
            The options are:
            False - do not create a plot (default);
            True - create a plot.
    var_names : list
        a list containing names of the nodes for the plot
    
    Returns
    -------
    R : numpy.ndarray
        Rank Correlation Matrix
        
    """
    # Checking validity of the input dimensions
    if is_data:
        if data.shape[1] != len(parent_cell):
            raise argparse.ArgumentTypeError('Number of data columns does not match the number of parent cells')
            # Reading the number of variables
        N = data.shape[1]

    if not is_data:
        if len(data) != len(parent_cell):
            raise argparse.ArgumentTypeError('Number of data columns does not match the number of parent cells')
            # Reading the number of variables
        N = len(data)

    # Constructing a valid 'sampling order', which means that the node with no 
    # parents will be the first in the sampling order (SO) and so forth.
    SO = []
    while len(SO) < N:
        # Storing the elements in [0:N] not contained in [sampling_order]
        # (using auxiliary function 9)
        indices = list_dif(range(N), SO)
        for i in indices:
            # qq is empty if the parents of i are already contained in SO
            qq = list_dif(parent_cell[i], SO)
            if not qq:
                # in case qq is empty, adding i to the sampling order
                SO.append(i)

    # Creating a data matrix out of specified rank correlation matrix, if such
    # input was chosen. 
    if not is_data:
        import copy
        data_r_to_p = copy.deepcopy(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                # Transforming Spearman's into Pearson's correlation 
                # (auxiliary function 1)
                data_r_to_p[i][j] = ranktopearson(data_r_to_p[i][j])

    # Transforming the data to standard normal (auxiliary function 7)
    if is_data:
        norm_data, _ = NormalTransform(data)

    # Initializing the correlation matrix R
    R = np.zeros((N, N))
    np.fill_diagonal(R, 1)

    # Initializing a cell containing information during the recursive
    # calculation of the correlation matrix given the network 
    # (see PartCorr function (3))
    L = [[None] * (N) for i in range(N)]

    # Starting the loop for recursively calculating the correlation matrix by
    # the second node
    for i in range(1, N):
        # Variables for the looping
        cond = []  # Vector storing the conditionalized variables
        T = np.zeros(i)  # Vector of the same length of previous nodes
        counter = 0  # Counter for recursive estimation

        seq = parent_cell[SO[i]]  # Contains the parent of the i-th node
        seq2 = list_dif(SO[0:i], seq)  # Contains the previous (same order of
        # SO!) nodes that are not parents

        # Looping over the parents of the i-th node
        for j in seq:
            if not is_data:
                T[counter] = data_r_to_p[SO[i]][len(cond)]
            else:
                # Calculating the partial correlation between the node 
                # (normdata(:,SO(i))) at its parent (normdata(:,j)) given the
                # conditioning variable(s) (normdata(:,cond))
                T[counter] = pg.partial_corr(data=norm_data,
                                             x=norm_data.columns[SO[i]],
                                             y=norm_data.columns[j],
                                             covar=list(norm_data.columns[i]
                                                        for i in cond),
                                             method='pearson').r.values[0]
            s = T[counter]

            # Looping over all the remaining parents
            for k in range(len(cond) - 1, -1, -1):
                # Recursivelly calculating the correlation between nodes 
                # accounting for the conditional/partial correlation 
                # established by the BN (auxiliary function 3)
                [L, r1] = PartCorr(j, cond[k], cond[0:k], R, L, N + 1)
                # Based on the conditional/partial correlation, calculating the
                # resulting correlation coefficient (all the properties of the 
                # correlation matrix are guaranteed)
                shat = s * np.sqrt((1 - T[k] ** 2) * (1 - (r1) ** 2)) + T[k] * r1
                s = shat
            if np.isnan(s):
                R[SO[i], j] = s
                R[j, SO[i]] = s
                print('Error, s = nan')
                print('j= ' + str(j) + ', k= ' + str(k))
            else:
                # Saving the correlation coefficients calculated in the upper 
                # and lower triangle of the matrix R.
                R[SO[i], j] = s
                R[j, SO[i]] = s
            counter += 1
            if not cond:
                cond = [j]
            else:
                cond.append(j)

        # Looping over the previous nodes (based on the ordering in SO) which 
        # are not parents, stored in seq2.
        for j in seq2:
            T[counter] = 0
            s = T[counter]
            for k in range(len(cond) - 1, -1, -1):
                if T[k] != 0 or s != 0:
                    # Recursively calculating the correlation between nodes 
                    # accounting for the conditional/partial correlation 
                    # established by the BN (auxiliary function 3)
                    L, r1 = PartCorr(j, cond[k], cond[0:k], R, L, N + 1)
                    # Based on the conditional/partial correlation, calculating 
                    # the resulting correlation coefficient (all the properties 
                    # of the correlation matrix are guaranteed)
                    shat = s * np.sqrt((1 - T[k] ** 2) * (1 - (r1) ** 2)) + T[k] * r1
                    s = shat

            # Storing the results
            R[SO[i], j] = s
            R[j, SO[i]] = s
            counter += 1
            if not cond:
                cond = [j]
            else:
                cond.append(j)
    # Transforming Pearson's correlation into Spearman's rank correlation 
    # (auxiliary function 2)
    R = pearsontorank(R)
    if plot:
        rank_corr_mat_fig(var_names, R)

    return R


# ----------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# ----------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------
# 1 - Transforming Spearman's rank correlation into Pearson's correlation
# ----------------------------------------------------------------------------
def ranktopearson(R):
    """
    ranktopearson transforms Spearman's rank correlation R into 
    Pearson's correlation r and returns r 

    Parameters
    ----------
    R : float
        Spearman's rank correlation
    
    Returns
    -------
    r : float
        Pearson's correlation
    """

    # Transforming rank correlation (R) into Pearson's correlation (r)
    r = 2 * np.sin((np.pi / 6) * R)
    return r


# ----------------------------------------------------------------------------
# 2 - Transforming Pearson's correlation into Spearman's rank correlation
# ----------------------------------------------------------------------------
def pearsontorank(r):
    """
    pearsontorank transforms Pearson's correlation r into 
    Spearman's rank correlation R and returns R

    Parameters
    ----------
    r : float
        Pearson's correlation 
    
    Returns
    -------
    R : float
        Spearman's rank correlation
    """

    # Transforming Pearson's correlation (r) into rank correlation (R)
    R = (6 / np.pi) * np.arcsin(r / 2)
    return R


# ----------------------------------------------------------------------------
# 3 - Calculating the partial correlation with a recursive approach 
# (different from the standard Matlab function partialcorr).
# ----------------------------------------------------------------------------
def PartCorr(i, j, cond, R, L, m):
    """
    PartCorr calculates the partial correlation with a recursive approach
    and returns both the partial correlations r and a list L containing 
    information about the partial correlations during the recursive calculation
    
    Parameters
    ----------
    i : int
        row index R to calculate the correlation
    j : int
        column index R to calculate the correlation
    cond : list
        conditioning variable(s)
    R : numpy.ndarray
        correlation matrix (partially filled-in)
    L : list
        a list with information about estimates from previous steps
    m : int
        the number of nodes + 1
    
    Returns
    -------
    L : list
        list of lists L with partial correlations from previous steps
    r : float
        partial correlation r

    """

    # Ordering of the indices
    s = np.sort([i, j])
    isort = s[0]
    jsort = s[1]

    # Defining the number of conditioning variables
    n = len(cond)

    # If the conditioning variable vector is empty, then the value is
    # obtained from the correlation matrix    
    if n == 0:
        r = R[i, j]
        return L, r

    # Extracting information on the cell L (from previous calculations)
    Lc = L[isort][jsort]

    # Calculating the index with auxiliary functions (4) and (6)
    if Lc:
        # If Lc has only one record, list has only one dimension, see except
        try:
            index = search_([item[0] for item in Lc], calc_val(cond, m))
        except IndexError as e:
            # Deal with exception
            index = search_([Lc[0]], calc_val(cond, m))
        if index:
            r = Lc[index][1]
            return L, r

    [L, r1] = PartCorr(i, j, cond[1:n], R, L, m)
    [L, r2] = PartCorr(i, cond[0], cond[1:n], R, L, m)
    [L, r3] = PartCorr(j, cond[0], cond[1:n], R, L, m)

    # Calculating partial correlation of [(i,j) | (cond(1),cond(2:n))]
    r = (r1 - r2 * r3) / ((1 - (r2) ** 2) * (1 - (r3) ** 2)) ** (0.5)

    # Saving the results (auxiliary functions 4 and 5)
    L[isort][jsort] = add_corr(L[isort][jsort], calc_val(cond, m), r)

    return L, r


# ----------------------------------------------------------------------------
# 4 - Calculating an index for the recursive procedure in PartCorr
# ----------------------------------------------------------------------------
def calc_val(cond, m):
    """
    calc_val calculates an index for the recursive procedure in PartCorr
    
    Parameters
    ----------
    cond : list
        conditioning variable(s) from auxiliary function 3
    m : int 
    the number of nodes + 1
    
    Returns
    -------
    v : int
        value of the index
    """
    n = len(cond)
    sc = np.sort(cond)
    if (n == 0):
        v = 0
        return v

    v = 0
    for i in range(0, n):
        v = (v + (sc[i] + 1) * m ** i)
    return v


# ----------------------------------------------------------------------------
# 5 - Adding correlation to cell L
# ----------------------------------------------------------------------------
def add_corr(Lc, val, r):
    """
    calc_val adds a correlation to list Lc and returns Lc with that addition
    
    Parameters
    ----------
    Lc : list 
        part of list L in which a correlation will be added
    val : int
        value of the index from auxiliary function 4
    r : float
        partial correlation from auxiliary function 3
    
    Returns
    -------
    Lc : list 
        new Lc with new correlation added
    """

    if not (Lc):
        Lc = [val, r]
        return Lc
    # If Lc has only one record, list has only one dimension, e.g. [2,0.378]
    try:
        if np.issubdtype(Lc[0], np.integer):
            Lc1 = [Lc[0]]
        # Else: get all first elements from a list like [[0,0,257],[2,0.378]]
        else:
            Lc1 = [item[0] for item in Lc]
    except TypeError as e:
        Lc1 = [item[0] for item in Lc]

    n1 = 0
    n2 = len(Lc1) + 1

    while ((n2 - n1) >= 2):
        n = int(np.ceil((n1 + n2) / 2))
        if (Lc1[n - 1] == val):
            index = n - 1
            return Lc
        else:
            if (Lc1[n - 1] < val):
                n1 = n
            else:
                n2 = n

    # If Lc has only one record, list has only one dimension, e.g. [2,0.378]
    try:
        if np.issubdtype(Lc[0], np.integer):
            Lc = [Lc]
    except TypeError as e:
        # Catch if Lc has more than one dimension (i.e. more than one elements)
        pass

    Lc.insert(n1, [val, r])
    return Lc


# ----------------------------------------------------------------------------
# 6 - A search function for the recursive procedure in PartCorr
# ----------------------------------------------------------------------------
def search_(Lc, val):
    """
    search_ searches an index for the recursive procedure in PartCorr
    and returns this index
    
    Parameters
    ----------
    Lc :  list
        value of L from previous calculations
    val : int
        value of the index from auxiliary function 4
    
    Returns
    -------
    index : int
        an index for the recursive procedure
    """

    n1 = 0
    n2 = len(Lc) - 1
    while n1 <= n2:
        n = np.floor((n1 + n2) / 2).astype('int')
        if Lc[n] == val:
            index = n
            return index
        else:
            if Lc[n] < val:
                n1 = n + 1
            else:
                n2 = n - 1
    index = 0
    return index


# ----------------------------------------------------------------------------
# 7 - Transforming data into ranked (uniform) and standard normal distribution
# ----------------------------------------------------------------------------
def NormalTransform(data):
    """
    NormalTransform transforms data into ranked (uniform) and 
    standard normal distribution
    
    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        input data to be transformed
    
    Returns
    -------
    norm_data : pandas.core.frame.DataFrame
        ranked standard normal distribution
    u_hat : pandas.core.frame.DataFrame
        ranked uniform distribution
    """

    M = data.shape[0]  # Reading number of observations per node
    ranks = data.rank(axis=0)
    u_hat = ranks / (M + 1)
    norm_data = u_hat.apply(np.vectorize(st.norm.ppf))
    norm_data.replace([np.inf, -np.inf], 0, inplace=True)  # Adjusting abnormal values
    return norm_data, u_hat


# ----------------------------------------------------------------------------
# 8 - Plot Rank Correlation matrix
# ----------------------------------------------------------------------------

def rank_corr_mat_fig(NamesBN, RBN):
    """
    rank_corr_mat_fig plots the Rank Correlation matrix
    
    Parameters
    ----------
    NamesBN : list
        BN nodes names from model.
    RBN : numpy.ndarray
        Correlation matrix.

    Returns
    -------
    None.

    """
    nam = NamesBN

    # Replace 0 with NaN
    z = np.round(RBN, 4)
    z = z.astype('float')
    z[z == 0] = 'nan'  # or use np.nan

    # reduce the names of the variables if the number of variables is large
    # to plot the only the reduced names
    if len(nam) > 80:
        nv = int(round(len(nam) / 26, 0))
    elif len(nam) <= 80 and len(nam) > 40:
        nv = int(round(len(nam) / 13, 0))
    elif len(nam) <= 40 and len(nam) > 20:
        nv = int(round(len(nam) / 21, 0))
    elif len(nam) <= 20:
        nv = len(nam)

    px = list(range(len(nam)))  # position of the labels

    fig, ax = plt.subplots(figsize=(15, 11))
    im = plt.imshow(z, cmap='Blues', interpolation='nearest')
    plt.colorbar()

    if len(nam) <= 20:
        plt.xticks(px, nam, rotation=45)
        plt.yticks(px, nam)
    else:  # plot the only the reduces names so the plot doesnt look saturated
        plt.xticks(px[::nv], nam[::nv], rotation=45)
        plt.yticks(px[::nv], nam[::nv])
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    if len(nam) <= 10:
        zz = np.round(RBN, 4)
        zz = zz.astype(str)
        for i in range(len(nam)):
            for j in range(len(nam)):
                if zz[i, j] == '0.0':
                    zz[i, j] = ''
                text = ax.text(j, i, zz[i, j],
                               ha="center", va="center",
                               color="k", fontsize='xx-large', fontweight='roman')
    plt.show()

    return None


# ----------------------------------------------------------------------------
# 9 - Subtract two lists
# ----------------------------------------------------------------------------

def list_dif(li1, li2):
    """
    list_dif subtracts two lists (li1-li2)

    Parameters
    ----------
    li1: list
        base list
    li2: list 
        list to be subtracted from base list
      
    Returns
    -------
    li1-li2: list
        li1-li2 as a new list
        
    """
    return list(list(set(li1) - set(li2)))
