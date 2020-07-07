# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 23:20:08 2020

@author: Amir Hesam
"""


#from sklearn import preprocessing as pr
from sklearn import datasets as de
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_curve, roc_auc_score
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

btc = de.load_breast_cancer()
#btc = de.load_iris()

x = btc.data
y = btc.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42, stratify=y)

log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
y_pred_prob = log.predict_proba(x_test)[:,1]

fpr, tpr, treshold = roc_curve(y_test, y_pred_prob)
roc_auc_score(y_test, y_pred_prob)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)