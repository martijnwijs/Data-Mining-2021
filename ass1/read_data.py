import pandas as pd
import numpy as np 
import csv 
from datetime import datetime
import itertools

# def create_daily_df(df): 
#     """
#     This function aggregates the variable values per day, where it takes the average of mood, arousal and valence. 
#     It takes the sum of the other variables. 
#     """

#     # Add date column to dataframe 
#     df["date"] = df["time"].str[:10]

#     # Define the variables of which we want the mean per day - all other variables are aggregated in a sum. 
#     mean_vars = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]

#     # Create the dataframe with the means of variables per day  
#     df_means = df[df["variable"].isin(mean_vars)]
#     df_means =  df_means.groupby(["id", "date", "variable"], as_index=False)[["value"]].mean()

#     # Create the datafram with the sums of variables per day
#     df_sums = df[~df["variable"].isin(mean_vars)]
#     df_sums = df_sums.groupby(["id", "date", "variable"], as_index=False)[["value"]].sum()

#     # Concatenate the two dataframes into the df with daily values
#     df_daily = pd.concat([df_sums, df_means])
#     df_daily = df_daily.sort_values(by=["id", "date"])

#     return df_daily

def create_daily_df(df): 
    """
    This function aggregates the variable values per day, where it takes the average of mood, arousal and valence. 
    It takes the sum of the other variables. 
    """

    # Add date column to dataframe 
    df["date"] = df["time"].str[:10]

    # Define the variables of which we want the mean per day - all other variables are aggregated in a sum. 
    mean_vars = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]

    # Create the dataframe with the means of variables per day  
    df_means = df[df["variable"].isin(mean_vars)]
    df_means =  df_means.groupby(["id", "date", "variable"], as_index=False)[["value"]].mean()

    # Create the datafram with the sums of variables per day
    df_sums = df[~df["variable"].isin(mean_vars)]
    df_sums = df_sums.groupby(["id", "date", "variable"], as_index=False)[["value"]].sum()
    
    # Create the dataframe with the frequency of screen 
    df_freq = df[df["variable"]=="screen"]
    df_freq = df_freq.groupby(["id", "date", "variable"], as_index=False)[["value"]].count()
    df_freq["variable"] = "frequency"

    # Concatenate the two dataframes into the df with daily values
    df_daily = pd.concat([df_sums, df_means, df_freq])
    df_daily = df_daily.sort_values(by=["id", "date"])

    return df_daily

def create_date_list(df_daily): 

    # Create a list of patient ids 
    ids = df_daily["id"].drop_duplicates().tolist()

    date_instances = []
    date_targets = []

    # Loop through the patient ids and create a list of lists of dates which will later correspond to a data instance. 
    for patient in ids: 
        
        date_instances_id = []
        date_targets_id = []
        
        # Get list of all dates that have a mood recording 
        date_strs = df_daily[(df_daily["variable"]=="mood") & (df_daily["id"] == patient)]["date"].tolist()
        
        # convert dates list to actual times
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in date_strs]
        
        # range through all dates of of a patient 
        for i in range(len(dates)-5):
            diff_time = dates[i+5] - dates[i]
            if diff_time.days != 5: 
                # To ensure there are 5 consecutive days 
                continue 
            date_instances_id.append(date_strs[i:i+4])
            date_targets_id.append(date_strs[i+4])
        
        date_instances.append(date_instances_id)
        date_targets.append(date_targets_id)

    return date_instances, date_targets 


def create_dataframe(df): 
    df_daily = create_daily_df(df)
    date_instances, date_targets = create_date_list(df_daily)
    ids = df_daily["id"].drop_duplicates().tolist()    
    variables = df_daily["variable"].drop_duplicates().tolist() 

    # Initialize 
    x_no = 0
    list_of_rows = []

    for (idx, id_dates, id_targets) in zip(ids, date_instances, date_targets): 
        for (dates, target_date) in zip(id_dates, id_targets): 
            x_no +=1
            df_dates_id = df_daily[(df_daily["date"].isin(dates)) & (df_daily["id"] == idx)]
            for t, date in enumerate(dates): 
                
                # Create dataframe based on the patient id and 
                df_t = df_dates_id[df_dates_id["date"]==date]
                
                # Get a list of variable names 
                names = df_t[df_t["variable"].isin(variables)]['variable'].tolist()
                names_missing = list(set(variables)-set(names))
                names_all = names + names_missing
                
                # Get list of variable values
                values = df_t[df_t["variable"].isin(variables)]['value'].tolist()
                if "screen" in names_missing: 
                    
                    # the average for that person 
                    imputations = []
                    for var in names_missing: 
                        others = df_daily[(df_daily["id"]==idx) & (df_daily["variable"]==var)] 
                        if len(others)==0: 
                            imputation = 0
                        else: 
                            imputation = others.mean()
                        imputations.append(float(imputation))
                    values_all = values + imputations
                    
                else:
                    values_all = values + len(names_missing) * [0]
                
                # Sort values based on the names_all list 
                values_sorted = [val for name,val in sorted(zip(names_all,values_all))]
                target = float(df_daily[(df_daily["date"]==target_date) & (df_daily["id"]==idx) & (df_daily["variable"]=="mood")]["value"])
                target = round(target)
                
                # Create list with: [data_instance_no, target, t, activity, appCat.builtin, ...]
                row = [x_no, target, t] + values_sorted 
                list_of_rows.append(row)

    columns = ["no", "target", "t"] + sorted(variables)
    dataframe = pd.DataFrame(list_of_rows, columns=columns)

    return dataframe 
# # Read dataset as pandas df
dataset = "dataset_mood_smartphone.csv" 
df = pd.read_csv(dataset)
df = df.fillna(method='ffill')

# Create new dataframe 
dataframe = create_dataframe(df)

# Save dataframe 
dataframe.to_csv("df_imp.csv", index=False)