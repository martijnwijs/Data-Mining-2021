import numpy as np 
import csv 

dataset = "dataset_mood_smartphone.csv" 

def read_dataset(dataset): 
    """ 
    This function reads the data into a list of dictionaries where: 
    [{id: AS14.01, time: }]
    """
    with open(dataset) as csv_file: 
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        ids = []
        list_of_dict =  [{} for i in range(40)]
        for row in csv_reader: 
        
            # get the id number from id
            id_no = int(row[1][-2:]) 
            # append to dictionary: "variable":(time, number) 
            if list_of_dict[id_no].get(row[3]) == None: 
                list_of_dict[id_no][row[3]] = [(row[2], row[4])]
            else: 
                list_of_dict[id_no][row[3]].append((row[2], row[4]))

    return list_of_dict 

data_list = read_dataset(dataset)
            
# def read_data_daily(dataset): 

#     with open(dataset) as csv_file: 
#         csv_reader = 

print(data_list[1])
