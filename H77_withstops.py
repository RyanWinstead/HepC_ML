import csv

#DRASTIC CHANGE FEATURE: ASSEMBLE DICTIONARY OF AA TYPES
poscharge = ['R','H','K']
negcharge = ['D','E']
uncharge = ['S','T','N','Q']
hydro= ['A','I','L','F','M','W','Y','V']
special = ['C','G','P']
classes = [poscharge, 'negcharge', 'uncharge','hydro', 'special']
AAdict = {}
for i in poscharge:
    AAdict.update({i:'poscharge'})
for i in negcharge:
    AAdict.update({i:'negcharge'})
for i in uncharge:
    AAdict.update({i:'uncharge'})
for i in hydro:
    AAdict.update({i:'hydro'})
for i in special:
    AAdict.update({i:'special'})
#print(AAdict)

#WRITES 'Drastic' COLUMN TO FILE'
with open("H77_drastic.csv", 'w', newline='') as file:
    w = csv.writer(file)
    v = open("/Users/ryanwinstead/Documents/GitHub/HepC_ML/H77_withstops.csv")
    r = csv.reader(v)
    row0 = next(r)
    row0.append("Drastic") 
    w.writerow(row0)

    for row in r:
        if row[3]== 'syn':
            row.append("0")
            w.writerow(row)
            continue
        else:
            wild = row[4]
            mut = row[5]
            if AAdict.get(wild) == AAdict.get(mut):
                row.append("0")
            else:
                row.append("1")
            w.writerow(row)

#WRITES "Stop" COLUMN TO FILE
'''
with open("H77_withstops.csv", 'w', newline='') as file:
    w = csv.writer(file)
    v = open("/Users/ryanwinstead/Documents/GitHub/HepC_ML/H77_metadata.csv")
    r = csv.reader(v)
    row0 = next(r)
    print(row0)
    row0.append("Stop") 
    w.writerow(row0)

    for row in r:
        if row[3] == "stop":
            row.append("1")
        else: 
            row.append("0")
        w.writerow(row)
    
    



for row in r:
    if row[3]== 'syn':
        continue
    else:
        wild = row[4]
        mut = row[5]
        if array(wild) == array(mut):

'''
