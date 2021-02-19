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



directory = r'C:\Users\ablej\OneDrive\Email attachments\Documents\Fall 2020\CSC 306 machine learning\Final'
AA_change = pd.read_csv('H77_metadata.csv')
sample_freqs = pd.read_csv('HCV1a_TsMutFreq_195.csv')
mut_rate = pd.read_csv('Geller.mutation.rates_update.csv')
allfeatures = pd.read_csv('allfeatures.csv')
logcosts = pd.read_csv('logCosts.csv')
onehot = pd.read_csv('allfeatures_num_OH.csv')

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
mut_AA = allfeatures['MutAA']
og_AA = allfeatures['WTAA']
"AA change into hydrophobic: alanine(A), isoleucine(I), valine(V), leucine(L), methionine(M), phenylalanine(F), Tyrosine(Y), Tyrptophan(W)"
hydrophobic = ['A','I','V','L','M','F','Y','W']
"AA change into polar: Asparagine(N), Glutatmine(Q), Tyrosine(Y), Serine(S), Threonine(T)"
polar = ['N','Q','Y','S','T']
"AA change into nonpolar: Glycine(G), Alanine(A), Valine(V), Leucine(L), Isoleucine(I), Tryptophan(W), Proline(P), Cysteine(C), Methionine(M), Phenyalanine(F)"
nonpolar = ['G','A','V','L','I','W','P','C','M','F']
acidity = ['G','A','V']

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

"Get histogram plot of data"
def gethistogram(datacolumn):
    ctlst=[] #list holding costs
    for i in datacolumn:
        ln = m.log(i) #take the natural log of numbers in costs, b/c cost values are really small
        ctlst.append(ln)
    ctlst.sort() #don't know if this is necessary
    plt.hist(ctlst, bins = 92) #bins determined by square root of total data points
    plt.ylabel('Frequency')
    plt.xlabel('ln(Cost for each position)')
    plt.title('Histogram of Costs for each positional mutation')
    plt.show()

"Run the Functions"
posChange = getfeatures(pos_AA,'Positive AA')
negChange = getfeatures(neg_AA,'Negative AA')
hydrophobicChange = getfeatures(hydrophobic,'Hydrophobic AA')
polarChange = getfeatures(polar,'Polar AA')
nonpolarChange = getfeatures(nonpolar,'Nonpolar AA')
acidtyChange = getfeatures(acidity,'pH AA')    
allfeaturesV2 = pd.concat([allfeatures,posChange,negChange,hydrophobicChange,polarChange,nonpolarChange,acidtyChange],axis=1, sort=False)
allfeaturesV2.to_csv ('allfeaturesV2.csv', index = False, header=True)
#costs = allfeatures['Cost']
#gethistogram(costs)

'''labels=[]
for num in onehot['Cost']:
    ln = m.log(num)
    if ln > -8:
        labels.append('High')
    elif ln <= -8:
        labels.append('Low')
labels_df = DataFrame(labels, columns=['Cost Label'])
"Make Tree"
y = labels_df
cpg = onehot['makesCpG']
otherfeats = onehot.iloc[:,6:]
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
#tree.plot_tree(specificTree, feature_names=x.columns,class_names=cn,filled = True)
#fig.savefig('test.png')


Sscore = specificTree.score(test_inputs, test_classes)
print(Sscore)

#specificTree_data = tree.export_graphviz(specificTree, feature_names = list(x.columns),class_names =cn ) 
#specific_graph = graphviz.Source(specificTree_data)

"Tree depth is a measure of how many splits a tree can make before coming to a prediction"



counter = 0
for i in labels:
    if i == 'Low':
        counter+=1
print(counter)'''

      
    
        
        





