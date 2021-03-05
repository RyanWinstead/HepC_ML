#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:05:11 2021

@author: solis
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
import os

# all features file path
allfeat_file = "/Users/solis/Desktop/GitHub/HepC_ML/allfeatures_final.csv"
# sends file into specific folder
DIR_out = '/Users/solis/Desktop/GitHub/HepC_ML/Output_Files'

# exports output to a numbered csv file 
export_path = DIR_out +'allfeatures_mutFreq_below50%.csv'

#omit data [[50%]] and above & export output to new csv file

def omit_fifty(export_path):
    with open(allfeat_file, 'rb') as f:       
        df = pd.read_csv(f, delimiter=' ')#, names = ['pos','makesCpG','bigAAChange','Avg_Mutation_Freq','Mutation_Rate','Cost','Positive',
        df.to_csv(export_path)                                             # 'AA,Negative', 'AA,Hydrophobic', 'AA,Polar', 'AA,Nonpolar', 'AA,5','UTR','Core','E1','E2',
                                                      #'HVR1','NS1','NS2','NS3','NS4A','NS4B','NS5A','NS5B'])
        
        #df_costs = df.columns
        
        for col_name in df.columns: 
            print(col_name)
omit_fifty(export_path)


        
#directory = r'C:\Users\ablej\OneDrive\Email attachments\Documents\Fall 2020\CSC 306 machine learning\Final'
AA_change = pd.read_csv('H77_metadata.csv')
sample_freqs = pd.read_csv('HCV1a_TsMutFreq_195.csv')
mut_rate = pd.read_csv('Geller.mutation.rates_update.csv')
basicfeatures = pd.read_csv('basicfeatures.csv')
logcosts = pd.read_csv('logCosts.csv')
onehot = pd.read_csv('allfeatures_num_OH.csv')
FeatsBelow50 = pd.read_csv('allfeatures_mutFreq_below50%.csv')
allfeats = pd.read_csv('allfeatures_final.csv')
RNAstructure = pd.read_csv('RNAStructure_Conserved.csv')

#print(sample_freqs.head())
#print(mut_rate.head())

"Create Features"
Avg_freqs = sample_freqs['Avg_Mutation_Freq']
pos = sample_freqs['pos']
makesCpG = AA_change['makesCpG']
gene = AA_change['gene']
ref = AA_change['ref']
wt= AA_change['WTAA']
mutAA = AA_change['MutAA']
AAchange = AA_change['bigAAChange']
result = pd.concat([pos,makesCpG,gene,ref,wt,mutAA,AAchange,Avg_freqs], axis=1, sort=False)
#print(result.head())
#result.to_csv ('features_hepC.csv', index = False, header=True)

"ADD MUTATION RATE FEAT"
nucleotides = ['g', 'c', 'a', 't']
rates = [3.17*10**(-6),4.24*10**(-6),1.32*10**(-5),1.13*10**(-5)]
# g to a = 3.17..., a to g =1.32.., t(u) to c = 1.13.., c to t(u) == 4.24
rate_dict ={}
for nuc in range(len(nucleotides)):
    rate_dict[nucleotides[nuc]] = rates[nuc]
rates =[]

for nucleotide in result['ref']:
    if nucleotide in nucleotides:
        rates.append(rate_dict[nucleotide]) 
              
rates_df = DataFrame(rates, columns=['Mutation_Rate'])        
#print(rates_df)
#more_feats = pd.concat([result,rates_df], axis=1, sort=False)
#more_feats.to_csv ('features+rates.csv', index = False, header=True)

"ADD AMINO ACID FEATURES ()"
"positive to negative, vice versa, no charge. Pos to neg = Lysine(K), Argenine(R), histine(H) to aspartatic acid(B), glutamate(E)"
pos_AA = ['K','R','H']
neg_AA = ['B','E']
mut_AA = basicfeatures['MutAA']
og_AA = basicfeatures['WTAA']
"AA change into hydrophobic: alanine(A), isoleucine(I), valine(V), leucine(L), methionine(M), phenylalanine(F), Tyrosine(Y), Tyrptophan(W)"
hydrophobic = ['A','I','V','L','M','F','Y','W']
"AA change into polar: Asparagine(N), Glutatmine(Q), Tyrosine(Y), Serine(S), Threonine(T)"
polar = ['N','Q','Y','S','T']
"AA change into nonpolar: Glycine(G), Alanine(A), Valine(V), Leucine(L), Isoleucine(I), Tryptophan(W), Proline(P), Cysteine(C), Methionine(M), Phenyalanine(F)"
nonpolar = ['G','A','V','L','I','W','P','C','M','F']
acidic = ['D','E']
basic = ['R','H','K']

"Getfeatures used to create a dataframe of AminoAcid changes based on feature chosen"
def getfeatures(AAlist,ColumnName):
    listwithAAchange=[]
    for i in range(len(mut_AA)):
        if og_AA[i] in AAlist:
            listwithAAchange.append(0)
        elif mut_AA[i] in AAlist:
            listwithAAchange.append(1)
        else:
            listwithAAchange.append(0)
    feature_df = DataFrame(listwithAAchange, columns=[ColumnName])
    return feature_df    

"Create RNA Structures Feature"
length=[]
length1=[] 
start = RNAstructure['Start']
end = RNAstructure['End']
HepCdnaLength = basicfeatures['pos'][-1:] #should be about 8600ish
for i in HepCdnaLength: # couldn't loop through i directly so I had to add the value of x to a list then loop through that??
    length.append(i)
    #print(length)
for toalHepClength in range(length[0]): #looping through length of HepCDNAstrand and creating a new list where everyvalue is 0
    length1.append(0)
#print(len(length1))
for region in range(len(RNAstructure['Region'])): #loop through each region or row in RNAstructure file
    for s in range(len(length1)): #loop the indexes of total hepc DNA length then replace index values in length1 with 1 indicating there is an RNA structure at this index
        if start[region] <= s <= end[region]:
            length1[s]=1
RNAstruct_onehot = length1[basicfeatures['pos'][0]:] #features file starts at position 264 and not 0, so we are slicing from beginning of features file
RNAstruct_df = DataFrame(RNAstruct_onehot, columns=['RNAstructure'])
#RNAstruct_df.head()


"Get histogram plot of data"
def gethistogram(datacolumn):
    ctlst=[] #list holding costs
    for i in datacolumn:
        if i <= .5:
            ln = m.log(i) #take the natural log of numbers in costs, b/c cost values are really small
            ctlst.append(ln)
    ctlst.sort() #don't know if this is necessary
    plt.hist(ctlst, bins = 92) #bins determined by square root of total data points
    plt.ylabel('Frequency')
    plt.xlabel('ln(Cost for each position with AvgMut_Freq below 50%)')
    plt.title('Histogram of Costs for each positional mutation')
    plt.show()

"Create labels (targets) which are costs"
def CreateCostsLabels(costs):
    labels=[]
    for num in costs:
        ln = m.log(num)
        if ln >= -6:
            labels.append('Very High')
        if -6 > ln >= -7:
            labels.append('High')
        if -7 > ln >= -9.5:
            labels.append('Low')
        if ln < -9.5:
            labels.append('Very low')
    labels_df = DataFrame(labels, columns=['Cost Label'])
    return labels_df

"Run the Functions"
posChange = getfeatures(pos_AA,'Positive AA')
negChange = getfeatures(neg_AA,'Negative AA')
hydrophobicChange = getfeatures(hydrophobic,'Hydrophobic AA')
polarChange = getfeatures(polar,'Polar AA')
nonpolarChange = getfeatures(nonpolar,'Nonpolar AA')
acidChange = getfeatures(acidic,'Acidic AA')
basicChange = getfeatures(basic,'Basic AA')
#allfeaturesV2 = pd.concat([basicfeatures,posChange,negChange,hydrophobicChange,polarChange,nonpolarChange],axis=1, sort=False)
allfeaturesV3 = pd.concat([allfeats,RNAstruct_df,acidChange,basicChange],axis=1, sort=False)
CostLabelChange = CreateCostsLabels(allfeats['Cost'])
allfeaturesV4 = pd.concat([allfeaturesV3,CostLabelChange],axis=1, sort=False)

"Create csv file with all features"
#allfeaturesV2.to_csv ('allfeaturesV2.csv', index = False, header=True)
#allfeaturesV3.to_csv('allfeaturesV3.csv', index=False, header=True)
allfeaturesV4.to_csv('allfeaturesV4.csv', index=False, header=True)

#################################################################################

# all features file path
allfeat_file = "/Users/solis/Desktop/GitHub/HepC_ML/allfeatures_final.csv"
# sends file into specific folder
DIR_out = '/Users/solis/Desktop/GitHub/HepC_ML/Output_Files'

# exports output to a numbered csv file 
export_path = DIR_out +'allfeatures_mutFreq_below50%.csv'

#omit data [[50%]] and above & export output to new csv file

def omit_fifty(export_path):
    with open(allfeat_file, 'rb') as f:       
        df = pd.read_csv(f)
        for col_name in df.columns: 
            #cost = df["Cost"]
            below_fifty = df[df["Cost"] < .500]
            below_fifty.to_csv(export_path)  
            print(below_fifty)
omit_fifty(export_path)


#################################################################################




"Create histogram of Costs"
#costs = FeatsBelow50['Cost']
#gethistogram(costs)
