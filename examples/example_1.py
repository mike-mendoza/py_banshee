# -*- coding: utf-8 -*-
"""
Py_BANSHEE
Authors: Paul Koot, Miguel Angel Mendoza-Lugo, Dominik Paprotny,
         Elisa Ragno, Oswaldo Morales-Nápoles, Daniël Worm

E-mail:  m.a.mendozalugo@tudelft.nl & O.MoralesNapoles@tudelft.nl

"""

from py_banshee.rankcorr import bn_rankcorr
from py_banshee.bn_plot import bn_visualize
from py_banshee.prediction import inference,conditional_margins_hist


#%%Defining the variables of the NPBN

# Names of the variables (nodes)
names = ['W',       #Total vehicle weight (W)
         'AX1',     #First axle load (AX1)
         'AX2',     #Second axle load (AX2)
         'AX3']     #Third axle load (AX3)


N = len(names) 		# number of nodes

#Parametric distributions of the nodes
distributions = ['norm','genextreme','norm','genextreme']	
parameters = [[192.52,29.33],[0.16,54.48,10.96],[87.93,12.84],[0.085,41.53,8.95]]


#%%Defining the structure of the NPBN
ParentCell = [None]*N
ParentCell[0] = [1,2,3]
ParentCell[1] = []
ParentCell[2] = [1]
ParentCell[3] = [2]

#Defining the rank correlation matrix
RankCorr = [None]*N
RankCorr[0] = [0.83,0.9,0.86]
RankCorr[1] = []
RankCorr[2] = [0.6]
RankCorr[3] = [0.76]

#%%Conditional rank correlation matrix
R = bn_rankcorr(ParentCell,RankCorr,var_names=names,is_data=False, plot=True)
# The plot shows that total vehicle weight has stronger conditional correlations with all axles loads,
# The conditional correlations with the first and third axle load are weaker.

#%% bn_visualize - Plot of the Bayesian Network
bn_visualize(ParentCell,R,names,fig_name='NPBN_B3')
# The plot presents the BN with 4 nodes and 5 arcs, with the (conditional)
# rank correlations indicated on the arcs.

#%% INFERENCE - making inference with the BN model

#Definining input variables and options
condition_nodes = [0]       #conditionalized variables (node W)
condition_values = [400]    #conditionalized value of the node (node W)

F = inference(Nodes = condition_nodes,      #Nodes that will be conditionalized
              Values = condition_values,    #Information used to conditionalize the NPBN
              R=R,                          #The rank correlation matrix
              DATA=[],                      #No empirical data is provided
              SampleSize=1000,              #Number of samples drawn when conditionalizing the NPBN
              empirical_data=False,         #Nodes of the NPBN are parametric distributions
              distributions=distributions,  #Corresponig distributions names of the nodes
              parameters=parameters,        #Corresponfing parameters of the distributions
              Output='full')                #Conditional empirical distributions

#%%Un-conditional and conditinal marginal histograms
conditional_margins_hist(F,                                 #conditional empirical distributions
                        DATA = [],                          #No empirical data is provided
                        names = names,                      #Names of the variables (nodes)
                        condition_nodes = condition_nodes,  #Conditionalized nodes
                        empirical_data = False,                 #Nodes of the NPBN are parametric distributions
                        distributions=distributions,        #Corresponig distributions names of the nodes
                        parameters=parameters)              #Corresponfing parameters of the distributions

# The plot shows a comprasion of un-conditional and conditinal marginal histograms.
# The un-conditional marginal histograms are computed with random samples of the provided parametric distributions. 
# The conditional marginal histograms are computed with the outpuf of the INFERENCE function.

