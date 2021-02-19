# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:57:08 2020

@author: ofaye
"""

# Package Imports
import pandas as pd
from pandas import DataFrame

# Variables
directory = r'C:\Users\ofaye\Documents\CSC 307 Machine Learning\FINAL_PROJECT_FILES'
allfeatures = pd.read_csv('allfeatures.csv')
allfeaturesV2 = pd.read_csv('allfeaturesV2.csv')
gene = allfeatures['gene']
Charges = allfeatures['Charges']

"""
# Splitting charges column
charge_values = [1,-1,0]
charge_change = ['Change to Positive', 'Change to Negative', 'Change to Neutral']
charge_dict ={}

for chrge in range(len(charge_values)):
    charge_dict[charge_values[chrge]] = charge_change[chrge]
charge_change =[]

for chargetype in Charges:
    if chargetype in charge_values:
        charge_change.append(charge_dict[chargetype]) 

charges_df = DataFrame(charge_change, columns=['AA_Charge']) 

print(charges_df)
"""

# One hot encode data
one_hot_gene = pd.get_dummies(allfeatures['gene'])
print(one_hot_gene)
#one_hot_charges = pd.get_dummies(charges_df)

# Adding the one hot encoded features to csv
#added_feats = pd.concat([allfeatures,one_hot_gene,charges_df,one_hot_charges], axis=1, sort=False)
added_feats = pd.concat([allfeaturesV2,one_hot_gene], axis=1, sort=False)

# Saving csv
added_feats.to_csv ('one_hot_featuresV2.csv', index = False, header=True)









