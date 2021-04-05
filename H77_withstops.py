import csv





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
    
    
       
