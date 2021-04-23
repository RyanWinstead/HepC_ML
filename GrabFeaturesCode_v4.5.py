import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math as m
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import one_hot_features as o



#directory = r'C:\Users\ablej\OneDrive\Email attachments\Documents\Fall 2020\CSC 306 machine learning\Final'
AA_change = pd.read_csv('H77_metadata_AvgFreqs.csv')

#FeatsBelow50 = pd.read_csv('allfeatures_mutFreq_below50%.csv')
#allfeats = pd.read_csv('allfeatures_final.csv')
RNAstructure = pd.read_csv('RNAStructure_Conserved.csv')


"Create Features"
Avg_freqs = AA_change['Avg_Mutation_Freq']
pos = AA_change['pos']
makesCpG = AA_change['makesCpG']
gene = AA_change['gene']
ref = AA_change['ref']
og_AA= AA_change['WTAA']
mut_AA = AA_change['MutAA']
AAchange = AA_change['bigAAChange']
fromMeta = pd.concat([pos,makesCpG,AAchange,Avg_freqs], axis=1, sort=False)

"ADD MUTATION RATE FEAT"
nucleotides = ['g', 'c', 'a', 't']
rates = [3.17*10**(-6),4.24*10**(-6),1.32*10**(-5),1.13*10**(-5)]
# g to a = 3.17..., a to g =1.32.., t(u) to c = 1.13.., c to t(u) == 4.24
rate_dict ={}
for nuc in range(len(nucleotides)):
    rate_dict[nucleotides[nuc]] = rates[nuc] #assigns each nucleotide to a mutation rate
rates =[]

for nucleotide in ref:
    if nucleotide in nucleotides:
        rates.append(rate_dict[nucleotide]) 
              
rates_df = DataFrame(rates, columns=['Mutation_Rate'])        
#print(rates_df)
MetawithRates = pd.concat([fromMeta,rates_df], axis=1, sort=False)

"Create Cost Column (Our Target)"
def CreateCostValue(mutRate,frq): #use mutationRate column and frequency as parameters
    CostValues=[]
    for i in range(len(mutRate)):
      c = mutRate[i]/frq[i]
      CostValues.append(c)
    CostsV_df = DataFrame(CostValues, columns=['Costs'])
    return CostsV_df        
CostsV_df = CreateCostValue(MetawithRates['Mutation_Rate'],Avg_freqs)
FeatsandLabelV1 = pd.concat([MetawithRates,CostsV_df], axis=1, sort=False)
FeatsandLabelV1.head()

"ADD AMINO ACID FEATURES ()"
"positive to negative, vice versa, no charge. Pos to neg = Lysine(K), Argenine(R), histine(H) to aspartatic acid(B), glutamate(E)"
pos_AA = ['K','R','H']
neg_AA = ['B','E']
"AA change into hydrophobic: alanine(A), isoleucine(I), valine(V), leucine(L), methionine(M), phenylalanine(F), Tyrosine(Y), Tyrptophan(W)"
hydrophobic = ['A','I','V','L','M','F','Y','W']
"AA change into polar: Asparagine(N), Glutatmine(Q), Tyrosine(Y), Serine(S), Threonine(T)"
polar = ['N','Q','Y','S','T']
"AA change into nonpolar: Glycine(G), Alanine(A), Valine(V), Leucine(L), Isoleucine(I), Tryptophan(W), Proline(P), Cysteine(C), Methionine(M), Phenyalanine(F)"
nonpolar = ['G','A','V','L','I','W','P','C','M','F']
acidic = ['D','E']
basic = ['R','H','K']
mut_type =["nonsyn","syn"]


"Getfeatures used to create a dataframe of AminoAcid changes based on feature chosen"
Ft_names=[]
def getfeatures(AAlist,ColumnName):
    listwithAAchange=[]
    for i in range(len(mut_AA)):
        if og_AA[i] in AAlist:
            listwithAAchange.append(0)
        elif mut_AA[i] in AAlist:
            listwithAAchange.append(1)
        else:
            listwithAAchange.append(0)
    Ft_names.append(ColumnName)
    feature_df = DataFrame(listwithAAchange, columns=[ColumnName])
    return feature_df 

"Combine two features into 1" #For ex. Hydrophobic (AAchange) + E1 (gene section)
def CombineFeats(C1,C2): #C for columns
    print(len(C1))
    featL=[]
    for i in range(len(C1)): #loop through indexes of 1 column
        if C1[i] == 1: #if 1 (true) in column1
            if C1[i]==C2[i]: #and if 1 in column2
                featL.append(1) #then both columns are a match and we can append a true (1) to the featL list
            else:
                featL.append(0) #right?
        else:
            featL.append(0)
    print(len(featL))
    name = C1.name+'+'+C2.name
    feature_df = DataFrame(featL, columns=[name])
    return feature_df  

"Create RNA Structures Feature"
length=[]
length1=[]
posAslist=pos.values.tolist()
start = RNAstructure['Start']
end = RNAstructure['End']
HepCdnaLength = len(pos) #gets last value of position column
for toalHepClength in range(HepCdnaLength): #looping through length of HepCDNAstrand and creating a new list where everyvalue is 0
    length1.append(0)
#print(len(length1))
for region in range(len(RNAstructure['Region'])): #loop through each region or row in RNAstructure file
    for s in range(len(length1)): #loop the indexes of total hepc DNA length then replace index values in length1 with 1 indicating there is an RNA structure at this index
        if s in posAslist:
            if start[region] <= s <= end[region]:
                length1[s]=1
#RNAstruct_onehot = length1[pos[0]:] #features file starts at position 264 and not 0, so we are slicing from beginning of features file
RNAstruct_df = DataFrame(length1, columns=['RNAstructure'])
#RNAstruct_df.head()


"Create nonSynonomous mutation feature"
nonsyn =[]
for mut in AA_change['Type']:
    if mut == mut_type[0]:
        nonsyn.append(1)
    else:
        nonsyn.append(0)
nonsyn_df = DataFrame(nonsyn, columns=['Nonsyn'])
        
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

CostLabelChange = CreateCostsLabels(CostsV_df['Costs'])

allfeaturesV7 = pd.concat([FeatsandLabelV1,CostLabelChange,o.one_hot_gene,RNAstruct_df,nonsyn_df,posChange,negChange,hydrophobicChange,polarChange,nonpolarChange,acidChange,basicChange],axis=1, sort=False)

"Try to combine Features"
combinedFt_dfs=[]
for i in o.one_hot_gene: #11 is used b/c that's the index where the gene columns start in the allfeat dataframe. The first colon is for all the rows, the 2nd is to slice thr columns
    print(i)
    for f in Ft_names:
        feature_df=CombineFeats(allfeaturesV7[f],allfeaturesV7[i])
        combinedFt_dfs.append(feature_df)
CombinedFeatures=pd.concat(combinedFt_dfs,axis=1,sort=False)

allfeaturesV8 = pd.concat([allfeaturesV7,CombinedFeatures],axis=1, sort=False)

"Create csv file with all features"
allfeaturesV8.to_csv('allfeaturesV8.csv', index=False, header=True)


"Create histogram of Costs"
#costs = FeatsBelow50['Cost']
#gethistogram(costs)






