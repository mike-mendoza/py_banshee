# BANSHEE
This repository contains BANSHEE - A MATLAB Toolbox for Non-Parametric Bayesian Networks. 

The code is an update of the original version supporting SoftwareX paper: https://doi.org/10.1016/j.softx.2020.100588

Bayesian Networks (BNs) are probabilistic, graphical models for representing complex dependency structures. They have many applications in science and engineering. This toolbox implements a particularly powerful variant Non-Parametric BNs. The software allows for quantifying the BN, validating the underlying assumptions of the model, visualizing the network and its corresponding rank correlation matrix, and finally making inference with a BN based on existing or new evidence. The toolbox also includes some applied BN models published in recent scientific literature.

The src/ directory contains BANSHEE.mltbx file containing installer of the MATLAB toolbox.

The docs/ directory contains a quick start guide to the toolbox. Please consult the guide before using the toolbox.

This version (1.2) compared with published version supporting the SoftwareX publication (v1.1) has the following changes:
- Adds two new real-life models for predicting flood losses for the residential and commercial sectors;
- Updates the quick start guide;
- Corrects the description in "predict_coastal_erosion.m";
- Removes three .mat files which contained data alreary present in other .mat files;
- Updates references to the SoftwareX paper and other publications.
