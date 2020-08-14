# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:31:59 2019


"""
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error


def svr_model(x_features,y_features,x_test= None,y_test = None):

    svr_lin = SVR(kernel='linear', C=1e3)
    y_lin = svr_lin.fit(X_train_val[:10000,:], Y_train_val[:10000])
    mae = mean_absolute_error(y_lin,y_test) 
    print('mae: ',mae,'\n')
    
    return mae
   
    
    
def svc_model(x_features,y_features,x_test= None,y_test = None, C = 1):

    svc_lin = SVC(kernel='linear', C=C)
    svc_lin.fit(x_features, y_features)

    return svc_lin
