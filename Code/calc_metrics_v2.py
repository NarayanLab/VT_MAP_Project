# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:48:39 2019


"""

from sklearn.metrics import confusion_matrix
import numpy as np

def calc_classification_metrics(y_true,y_pred):
    conf_mat = confusion_matrix(y_true,y_pred)
    tn, fp, fn, tp = conf_mat.ravel()
    acc = (conf_mat[0,0]+conf_mat[1,1])/np.sum(conf_mat)
    sens = tp/(tp+fn)#(conf_mat[1,1])/(conf_mat[1,0]+conf_mat[1,1])
    spec = tn/(tn+fp)#(conf_mat[0,0])/(conf_mat[0,1]+conf_mat[0,0])
    ppv = tp/(tp+fp)#(conf_mat[1,1])/(conf_mat[0,1]+conf_mat[1,1])
    npv = tn/(tn+fn)#(conf_mat[0,0])/(conf_mat[0,0]+conf_mat[1,0])
    
    return conf_mat, acc, sens, spec, ppv, npv