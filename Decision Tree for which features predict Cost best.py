# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:16:11 2021

@author: ablej
"""
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math as m
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz 
import GrabFeaturesCode_v1 as G

"Create labels (targets) which are costs"
labels=[]
for num in G.onehot['Cost']:
    ln = m.log(num)
    if ln > -8:
        labels.append('High')
    elif ln <= -8:
        labels.append('Low')
labels_df = DataFrame(labels, columns=['Cost Label'])

"Make Tree"
y = labels_df
cpg = G.onehot['makesCpG']
otherfeats = G.onehot.iloc[:,6:]
x = pd.concat([cpg,otherfeats],axis=1,sort=False)
all_inputs = x.values
all_classes = y.values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)


dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
score = dtc.score(test_inputs, test_classes)

specificTree = tree.DecisionTreeClassifier(max_depth =5, min_samples_split = 5,min_samples_leaf = 4, max_features = 5)
specificTree.fit(train_inputs, train_classes)
cn = ['High','Low']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=500)
tree.plot_tree(specificTree, feature_names=x.columns,class_names=cn,filled = True)
#fig.savefig('test.png')


Sscore = specificTree.score(test_inputs, test_classes)
print(Sscore)

specificTree_data = tree.export_graphviz(specificTree, feature_names = list(x.columns),class_names =cn ) 
specific_graph = graphviz.Source(specificTree_data)