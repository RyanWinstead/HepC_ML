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

"READ ALL CSV FILES IN CSVS FOLDER"
os.chdir("../")
#os.getcwd()
directory = "../Github/HepC_ML/Csvs" #*YOUR DATA FOLDER in place of Csvs*
filedict = {}
for subdir, dirs, files in os.walk(directory):
    for file in files:
        filename = str(file)
        if filename == ".DS_Store":
            continue
        print(filename)
        path = (os.path.join(subdir, file))
        filedict[filename.rsplit('.', 1)[0]] = pd.read_csv(path)
    print(filedict.keys())
'''
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
"RYANS FEATURES"
helix_break = AA_change['Pro_Gly']

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

'''