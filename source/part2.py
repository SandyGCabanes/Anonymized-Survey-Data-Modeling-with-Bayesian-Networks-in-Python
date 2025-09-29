# %% [markdown]
# Originated by: Sandy G. Cabanes
# # This generates a synthetic dataset from the model with the lowest ess that achieves no floating nodes.  

# %%
import time
import numpy as np
import pandas as pd
import pgmpy
import pickle

#from pgmpy.estimators import HillClimbSearch, BDeu, BayesianEstimator,MaximumLikelihoodEstimator
#from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
# See part2.pdf for continuation
