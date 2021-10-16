# -*- coding: utf-8 -*-
"""
Py_BANSHEE
Authors: Paul Koot, Miguel Angel Mendoza-Lugo, Dominik Paprotny,
         Elisa Ragno, Oswaldo Morales-Nápoles, Daniël Worm

E-mail:  m.a.mendozalugo@tudelft.nl & O.MoralesNapoles@tudelft.nl


This code reproduce the non-parametric Bayesian network and the sample based conditioning  
    case number 6 of the article: Reliability analysis of reinforced concrete vehicle 
    bridges columns using non-parametric Bayesian networks 
    (https://doi.org/10.1016/j.engstruct.2019.03.011)

The autors used UNINET software to compute and present the results. 

Note: For this example the CVM_STATISTICS - test goodness-of-fit of the Gaussian copula
        Tested 153 copluas with over 3500 observations per variable.  
        ----- Finished in 1646.26 sec = 27.44  min --------
        in a machine wiht a Porcessor Intel Core i7-8665U 
        CPU @ 1.99GHz 2.11GHz, 32 GB  of installed memory RAM and Windows 10 operative system.
        
        Becouse of the number of copulas the tight_layout cannot make axes width small enough 
        to accommodate all axes decorations.
        

"""

from py_banshee.rankcorr import bn_rankcorr
from py_banshee.bn_plot import bn_visualize
from py_banshee.d_cal import gaussian_distance
from py_banshee.copula_test import cvm_statistic
from py_banshee.sample_bn import generate_samples
from py_banshee.sample_bn import sample_based_conditioning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#%%Defining the variables of the NPBN

data = pd.read_csv('Concrete_vehicle_bridge_column.csv')  #Reading dataset

# Select the columns to use in the NPBN
columns_used=[2,3,16,17,18,19,20,0,1,4,5,6,7,8,9,10,11,12]                             
data = data.iloc[:,columns_used] 

# Names of the variables (nodes)

# +--------+--------------------------------------------------+
# |  Node  |                    Description                   |
# +--------+--------------------------------------------------+
# | ApL1   | Number of axles in lane 1 of the bridge.         |
# +--------+--------------------------------------------------+
# | ApL2   | Number of axles in lane 2 of the bridge.         |
# +--------+--------------------------------------------------+
# | WA1    | Total vehicle weight in lane 1.                  |
# +--------+--------------------------------------------------+
# | WA2    | Total vehicle weight in lane 2.                  |
# +--------+--------------------------------------------------+
# | PGA    | Peak ground acceleration.                        |
# +--------+--------------------------------------------------+
# | f'c    | Compressive strength of concrete.                |
# +--------+--------------------------------------------------+
# | Ec     | Modulus of elasticity of the concrete.           |
# +--------+--------------------------------------------------+
# | fy     | Yield strength of steel.                         |
# +--------+--------------------------------------------------+
# | fu     | Tensile strength.                                |
# +--------+--------------------------------------------------+
# | MaxP   | Maximum Axial load.                              |
# +--------+--------------------------------------------------+
# | MaxV2  | Maximum Shear in direction 2.                    |
# +--------+--------------------------------------------------+
# | MaxV3  | Maximum Shear in direction 3.                    |
# +--------+--------------------------------------------------+
# | MaxM2  | Maximum Bending moment in direction 2.           |
# +--------+--------------------------------------------------+
# | MaxM3  | Maximum Bending moment in direction 3.           |
# +--------+--------------------------------------------------+
# | MaxT   | Maximum Torsional moment.                        |
# +--------+--------------------------------------------------+
# | U1     | Displacement of the study joint in direction 1.  |
# +--------+--------------------------------------------------+
# | U2     | Displacement of the study joint in direction 2.  |
# +--------+--------------------------------------------------+
# | U3     | Displacement of the study joint in direction 3.  |
# +--------+--------------------------------------------------+

names = list(data.columns) # ApL1, ApL2, PGA, fc, Ec, fy, fu, WA1, WA2, MaxP, MaxV2, MaxV3, MaxM2, MaxM3, MaxT, U1, U2, U3

N = data.shape[1] # number of nodes

#%%Defining the structure of the NPBN
parent_cell = [None]*N

parent_cell[0] = []           
parent_cell[1] = []          
parent_cell[2] = []       
parent_cell[3] = []           
parent_cell[4] = [3] 
parent_cell[5] = []
parent_cell[6] = [5]
parent_cell[7] = [0]
parent_cell[8] = [1]
parent_cell[9] = [2,3,4,5,6,7,8]
parent_cell[10] = [2,3,4,5,6,7,8,9]
parent_cell[11] = [2,3,4,5,6,7,8,9,10]
parent_cell[12] = [2,3,4,5,6,7,8,9,10,11]
parent_cell[13] = [2,3,4,5,6,7,8,9,10,11,12]
parent_cell[14] = [2,3,4,5,6,7,8,9,10,11,12,13]
parent_cell[15] = [2,3,4,5,6,7,8,9,10,11,12,13,14]
parent_cell[16] = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
parent_cell[17] = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#%% BN_RANKCORR - Conditional rank correlation matrix
R=bn_rankcorr(parent_cell,data,var_names=names,is_data=True, plot=True)
# The plot shows that displacements has stronger conditional correlations the bending moments 
# and the the peak ground aceleration.
# Negative conditional correlations are observed between the shear force and the bending moments, 
# the same is true with respect to the displacements.

#%% BN_VISUALIZE - Plot of the Bayesian Network
bn_visualize(parent_cell,R,data.columns,data=data,fig_name='NPBN_Bridge_Marginals')
bn_visualize(parent_cell,R,data.columns,fig_name='NPBN_Bridge')

# Two kinds of DAG can be ploted: i) A NPBN in which nodes are ellipses whit the name of the variabale 
# and ii) a NPBN with the marginals displayed inside of the nodes. Both DAGs with the (conditional)
# rank correlations indicated on the arcs.
#%% CVM_STATISTICS - test goodness-of-fit of the Gaussian copula* 
##M = cvm_statistic(data, plot=1, names=data.columns,fig_name='cvm_Bridge')
# The plot shows the 153 tested copluas with over 3500 observations per variable.
# *Time to compute over 20 min 
# The results of the goodness-of-fit test in terms of Cramer-von Mises
# statistic highlight that the Gaussian copula is in majority of cases the 
# most suitable for representing the dependency between variables. This is important as
# the method utilizes the Gaussian copula for dependence modelling.

#%% GAUSSIAN_DISTANCE - measuring d-calibration score
no_iterations = 1800                # Perform 1800 iterations 
SampleSize_ERC_NRC = 165            # draw 165 samples of the normal distribution 
                                    # and perform 1800 iterations to obtain the
                                    # distribution of the d-cal score for NRC test
SampleSize_NRC_BNRC = 1800          # draw 1800 samples of the normal distribution 
                                    # and perform 1800 iterations to obtain the
                                    # distribution of the d-cal score for BNRC test


D_ERC,B_ERC,D_BNRC,B_BNRC = gaussian_distance(R,                   # rank correlation matrix (function 1)
                                              data,                # matrix of data
                                              SampleSize_ERC_NRC,  # number of samples drawn for NRC test
                                              SampleSize_NRC_BNRC, # number of samples drawn for BNRC test 
                                              no_iterations,       # number of iterations
                                              Plot=True,           # create a plot (0 = don't create plot)
                                              Type='H',                 # type of distance metric (H = Hellinger distance)
                                              fig_name='d_cal_bridge')      # plot name
# The d-calibration score of the empirical rank correlation matrix is
# inside the 90% confidence interval of the determinant of the empirical
# normal distribution. The d-calibration 
# score of the BN's rank correlation matrix is well within the 90% 
# confidence interval of the determinant of the random normal distribution 
# sampled for the same correlation matrix. This supports the assumptions of
# a joint normal copula used in the BN model. It should be noted that the 
# test is sensitive to the number of samples drawn as well as the number of 
# iterations and is rather severe for large datasets.

#%% GENERATE SAMPLES -Compute N un-conditional samples from the non-parametric Bayes Network
# To obtain statistically robust results, we must increase the number of samples of the NPBN

no_samples = 1000000                                # no of samples to be generate
samples = generate_samples(R,no_samples,names,data)  
# 1000000 un-conditional samples for each variable are computed from the non-parametric Bayes Network 

#%% SAMPLE BASED CONDITIONING - Sample based conditioning conditionalizes on intervals.
# Sample based contiong case number 6 of Table 4 of the article: 
# Reliability analysis of reinforced concrete vehicle 
# bridges columns using non-parametric Bayesian networks 
 
input_nodes = [2,3,5,7,8]                          # Conditionalized variables (PGA,fc,fy,WA1,WA2)
output_nodes = list(list(set(list(                 # Output nodes (all other variables) 
                    range(N)))-set(input_nodes)))   

#Condtion intervals, CASE 6: PGA = Middle, WA1 = High, WA2 = High, fc = Low and fy = Low
lb_ub = [None]*len(input_nodes)
lb_ub[0] =(0.273, 0.783)    # Lower bound and upper bound of the PGA   
lb_ub[1] =(22.70, 30.00)    # Lower bound and upper bound of the fc
lb_ub[2] =(345.5, 435.0)    # Lower bound and upper bound of the fy
lb_ub[3] =(676.0, 1464.4)   # Lower bound and upper bound of the WA1
lb_ub[4] =(705.0, 1464.4)   # Lower bound and upper bound of the WA2

sm_bc = sample_based_conditioning(samples,input_nodes,lb_ub)
# The results are the sample based conditional empirical distributions of the output nodes.
# Oput nodes are those for which we wish to view the effects of conditionalization. 


#%% Comparing results Uninet - PyBanshee output 

#Compare Rank correlation matrix (compare determinants)
uninet_R = np.loadtxt('UNINET_BN_Rank_corr_mat.txt') # rank correlation matrix from uninet
uninet_determinant = np.linalg.det(uninet_R)         # determinant of the uninet rank correlation matrix
pybanshee_determinant = np.linalg.det(R)             # determinant of the PyBanshee rank correlation matrix

det_diff = abs(uninet_determinant-pybanshee_determinant) # absolute differnece between determinants
det_ratio = pybanshee_determinant/uninet_determinant     # rato between determinants
print('The difference between determinatns is '+ str(round(det_diff,11))+ ' and the ratio is ' + str(round(det_ratio,5)))


#Compare sample based conditioning: CASE 6
unitet_data_c6 = pd.read_csv('Samp_based_Case_6.csv')      # Sample based conditional samples from UNINET
unitet_data_c6 = unitet_data_c6[sm_bc.columns] # reorder the columns 

#Randomly remove data from the biggest sample to match sizes
to_remove = np.random.choice(sm_bc.index,size=len(sm_bc)-len(unitet_data_c6),replace=False)
sm_bc=sm_bc.drop(to_remove)

units = ['','','g','MPa','MPa','MPa','MPa','kN','kN','kN','kN','kN','kN.m','kN.m','kN.m','m','m','m']
#Empircal cumulative plots
def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

for i in range(len(output_nodes)):
    plt.semilogy(ecdf(sm_bc.iloc[:,output_nodes[i]])[0],1-ecdf(sm_bc.iloc[:,output_nodes[i]])[1],'.b', label='PY_BANSHEE')
    plt.semilogy(ecdf(unitet_data_c6.iloc[:,output_nodes[i]])[0],1-ecdf(unitet_data_c6.iloc[:,output_nodes[i]])[1],'.r', label='UNINET')
    plt.title(data.columns[output_nodes[i]])
    plt.xlabel(units[output_nodes[i]])
    plt.legend()
    plt.grid(axis='both')
    plt.savefig('ecdf_0'+str(i)+'.png')
    plt.show()
 
# The results shows that the difference between determinants is 4e-11 and the ratio is 1.00013
# The comparison of the empircal cumulative plots of the Sample based conditional samples from UNINET
# and the Sample based conditional samples cumputed by PY-BANSHEE are aling. It can be observe slighlty 
# differences in the output variables due to the randomness of sampling methods. 

