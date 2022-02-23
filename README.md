# BANSHEE

Bayesian Networks (BNs) are probabilistic, graphical models for representing complex dependency structures. They have many applications in science and engineering. Their particularly powerful variant – Non-Parametric BNs – are  implemented as an open-access scriptable code, in the form of a Python-based package.

This repository contains:
* BANSHEE - A MATLAB Toolbox for Non-Parametric Bayesian Networks.
* Py_BANSHEE a Python-based open source of the MATLAB toolbox.

These codes are an update of the original version supporting SoftwareX paper: https://doi.org/10.1016/j.softx.2020.100588

The packages allows for quantifying the BN, validating the underlying assumptions of the model, visualizing the network and its corresponding rank correlation matrix, sampling and finally making inference with a BN based on existing or new evidence. BANSHEE Matlab v1.3 and Py_BANSHEE have the same features.

# Py_BANSHEE

Py_BANSHEE  is a Python-based open source of the MATLAB toolbox [BANSHEE](https://doi.org/10.1016/j.softx.2020.100588). 

## Installation and updating
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Toolbox like below. 
Rerun this command to check for and install  updates .
```bash
pip install py-banshee
```

## Usage
Features:
* py_banshee.rankcorr.bn_rankcorr  --> Create a Bayesian Network rank correlation matrix 
* py_banshee.bn_plot.bn_visualize    --> Visualize the structure of a defined Bayesian Network
* py_banshee.copula_test.cvm_statistic      --> Goodness-of-fit test for copulas
* py_banshee.d_cal.gaussian_distance  --> Measure the distance between Gaussian densities
* py_banshee.sample_bn.generate_samples  --> Make samples using the defined Bayesian Network
* py_banshee.sample_bn.sample_base_conditioning --> Make samples based conditioning on intervals
* py_banshee.prediction.inference  --> Make predictions using a non-parametric Bayesian Network
* py_banshee.prediction.conditional_margins_hist  --> Visualize the un-conditional and conditional marginal histograms

#### Demo of some of the features:

```python
from py_banshee.rankcorr import bn_rankcorr
from py_banshee.bn_plot import bn_visualize
from py_banshee.prediction import inference,conditional_margins_hist

#Defining the variables of the BN
names = ['V1','V2','V3']  #names of the variables (nodes)
N = len(names) 		      #number of nodes

#parametric distributions of the nodes
distributions = ['norm','genextreme','norm']	
parameters = [[100,23],[-0.15,130,50],[500,100]]

#Defining the structure of the BN
ParentCell = [None]*N
ParentCell[0] = []
ParentCell[1] = [0]
ParentCell[2] = [0,1]

#Defining the rank correlation matrix
RankCorr = [None]*N
RankCorr[0] = []
RankCorr[1] = [.1]
RankCorr[2] = [.41,-.25]

#Conditional rank correlation matrix
R = bn_rankcorr(ParentCell,RankCorr,var_names=names,is_data=False, plot=True)

#Plot of the Bayesian Network
bn_visualize(ParentCell,R,names,fig_name='BN_TEST')

# Inference
condition_nodes = [0] #conditionalized variables (node V1)
condition_values = [181] #conditionalized value (node V1)

F = inference(Nodes = condition_nodes,
              Values = condition_values,
              R=R,
              DATA=[],
              SampleSize=100000,
              empirical_data=False, 
              distributions=distributions,
              parameters=parameters,
              Output='full')

#Conditional and un-conditional histograms 
conditional_margins_hist(F,[],names,condition_nodes,
                         empirical_data = False,
                         distributions=distributions,
                         parameters=parameters)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)
