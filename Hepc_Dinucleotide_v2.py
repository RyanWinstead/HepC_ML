# -*- coding: utf-8 -*-
"""
Description: Program that reads a csv file with a nucleotide column and determines 
whether a certain dinucleotide will occur if there is a mutation at a certain position

Version History:
    Version 1 - Determined dinucleotides based on ref and mut amino acids (incorrect version)
    Version 2 - Determined dinucleotides based on A/T or C/G mutation in ref nuc column (correct version)

Created on Fri Apr 23 2021

@author: Faye
"""

# IMPORTS
import pandas as pd

# STARTING DF VARIABLES
directory = r'C:\Users\ofaye\Documents\GitHub - Copy\HepC_ML'
H77_metadata = pd.read_csv('H77_metadata.csv')

#create previous and next nucleotide columns
H77_metadata['prev_nuc'] = H77_metadata['ref'].shift(+1)
H77_metadata['next_nuc'] = H77_metadata['ref'].shift(-1)
    
#lists to store each dinucleotide type
dinucA_column = []
dinucC_column = []
dinucG_column = []
dinucT_column = []

#method for the A dinuc
def a_dinuc():
    for index, row in H77_metadata.iterrows(): #iterrates through each row in the df
        if True: #set True for now, can be set False if you don't want to run this method
            if (row['ref'] == 'g' and row['prev_nuc'] == 'a') or (row['ref'] == 'g' and row['next_nuc'] == 'a'):
                dinucA_column.append('makesApA') #add to A dinucleotide list
            elif (row['ref'] == 't' and row['prev_nuc'] == 'a') or (row['ref'] == 'g' and row['next_nuc'] == 'c'):
                dinucA_column.append('makesApC')
            elif (row['ref'] == 'a' and row['prev_nuc'] == 'a') or (row['ref'] == 'g' and row['next_nuc'] == 'g'):
                dinucA_column.append('makesApG')
            elif (row['ref'] == 'c' and row['prev_nuc'] == 'a') or (row['ref'] == 'g' and row['next_nuc'] == 't'):
                dinucA_column.append('makesApT')
            else:
                dinucA_column.append('NA') #if the row is unable to make an A dinucleotide
    return dinucA_column

#method for the C dinuc
def c_dinuc():
    for index, row in H77_metadata.iterrows():
        if True:
            if (row['ref'] == 'g' and row['prev_nuc'] == 'c') or (row['ref'] == 't' and row['next_nuc'] == 'a'):
                dinucC_column.append('makesCpA')
            elif (row['ref'] == 't' and row['prev_nuc'] == 'c') or (row['ref'] == 't' and row['next_nuc'] == 'c'):
                dinucC_column.append('makesCpC')
            elif (row['ref'] == 'a' and row['prev_nuc'] == 'c') or (row['ref'] == 't' and row['next_nuc'] == 'g'):
                dinucC_column.append('makesCpG_2')
            elif (row['ref'] == 'c' and row['prev_nuc'] == 'c') or (row['ref'] == 't' and row['next_nuc'] == 't'):
                dinucC_column.append('makesCpT')
            else:
                dinucC_column.append('NA')
    return dinucC_column

#method for the G dinuc
def g_dinuc():
    for index, row in H77_metadata.iterrows():
        if True:
            if (row['ref'] == 'g' and row['prev_nuc'] == 'g') or (row['ref'] == 'a' and row['next_nuc'] == 'a'):
                dinucG_column.append('makesGpA')
            elif (row['ref'] == 't' and row['prev_nuc'] == 'g') or (row['ref'] == 'a' and row['next_nuc'] == 'c'):
                dinucG_column.append('makesGpC')
            elif (row['ref'] == 'a' and row['prev_nuc'] == 'g') or (row['ref'] == 'a' and row['next_nuc'] == 'g'):
                dinucG_column.append('makesGpG')
            elif (row['ref'] == 'c' and row['prev_nuc'] == 'g') or (row['ref'] == 'a' and row['next_nuc'] == 't'):
                dinucG_column.append('makesGpT')
            else:
                dinucG_column.append('NA')
    return dinucG_column

#method for the T dinuc
def t_dinuc():
    for index, row in H77_metadata.iterrows():
        if True:
            if (row['ref'] == 'g' and row['prev_nuc'] == 't') or (row['ref'] == 'c' and row['next_nuc'] == 'a'):
                dinucT_column.append('makesTpA')
            elif (row['ref'] == 't' and row['prev_nuc'] == 't') or (row['ref'] == 'c' and row['next_nuc'] == 'c'):
                dinucT_column.append('makesTpC')
            elif (row['ref'] == 'a' and row['prev_nuc'] == 't') or (row['ref'] == 'c' and row['next_nuc'] == 'g'):
                dinucT_column.append('makesTpG')
            elif (row['ref'] == 'c' and row['prev_nuc'] == 't') or (row['ref'] == 'c' and row['next_nuc'] == 't'):
                dinucT_column.append('makesTpT')
            else:
                dinucT_column.append('NA')
    return dinucT_column

#run methods and store them as columns in df
H77_metadata['dinucA'] = a_dinuc()
H77_metadata['dinucC'] = c_dinuc()
H77_metadata['dinucG'] = g_dinuc()
H77_metadata['dinucT'] = t_dinuc()

#one hot encode each dinuc column
one_hot_a_dinuc = pd.get_dummies(H77_metadata['dinucA'])
one_hot_c_dinuc = pd.get_dummies(H77_metadata['dinucC'])
one_hot_g_dinuc = pd.get_dummies(H77_metadata['dinucG'])
one_hot_t_dinuc = pd.get_dummies(H77_metadata['dinucT'])

#add one hot encoded columns to df 
added_a_nuc = pd.concat([H77_metadata, one_hot_a_dinuc, one_hot_c_dinuc, one_hot_g_dinuc, one_hot_t_dinuc], axis=1, sort=False)

#drop NA columns, dinucACGT columns, and extra CpG column so we only keep the makes_p_ columns
makesDinuc_only = added_a_nuc.drop(columns=['NA', 'dinucA', 'dinucC', 'dinucG', 'dinucT', 'makesCpG_2'])

#save df to csv
makesDinuc_only.to_csv('H77_metadata_dinuc_v2.csv', index=False, header=True)

"uncomment this code to get allfeaturesV8_MutFreqbelow50%"
#current_df = pd.read_csv('allfeaturesV7_MutFreqbelow50%.csv')
#merged_dinuc_df = pd.merge(current_df, makesDinuc_only, on='pos') #merge the two using 'pos' column
#drop_start_index = 29 #drop starting at column 29 which is the index column
#drop_end_index = 39 #drop ending at column 38 which is the next_nuc column
#finalized_df = merged_dinuc_df.drop(merged_dinuc_df.columns[drop_start_index:drop_end_index], axis=1) #drop desired columns
#finalized_df.to_csv('allfeaturesV8_MutFreqbelow50%.csv', index=False, header=True)


