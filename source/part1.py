# %% [markdown]
# Originated by: Sandy G. Cabanes
# September 9, 2025
# This script generates the Bayesian Network model using HillClimbSearch, BDeu scoring and fits the model with Bayesian Estimator and Maximum Likelihood Estimator

# %%
# Import packages
import time
import numpy as np
import pandas as pd
import pgmpy
import pickle
import os

from pgmpy.estimators import HillClimbSearch, BDeu, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork, DiscreteBayesianNetwork #For Naive Bayes seed
from pgmpy.sampling import BayesianModelSampling
# see part1.pdf for continuation
