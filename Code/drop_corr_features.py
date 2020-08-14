# -*- coding: utf-8 -*-
"""

"""
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error

# code is modified from original post by Chris Albon (https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/)

def drop_input_corr_columns(x_features, corr_fac = 0.90):
    # Create correlation matrix
    
    corr_matrix = x_features.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than corr_fac
    to_drop = [column for column in upper.columns if any(upper[column] > corr_fac)]
    
    # Drop features 
    x_features_no_colnr = x_features.drop(columns = to_drop)
    
    
    return x_features_no_colnr, to_drop
    