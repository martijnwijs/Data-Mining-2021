import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

def preprocess_dates(df):
    df.date_time = pd.to_datetime(df.date_time)
    df['year'], df['month'], df['day'], df['hour'], df['minute'] = df.date_time.dt.year, df.date_time.dt.month, df.date_time.dt.day, df.date_time.dt.hour, df.date_time.dt.minute
    return df


def rank_variable(df, variable):
    '''ranks on variable, way faster now because of groupby'''
    df_agg = df.groupby("srch_id")
    df_agg.apply(lambda _df: _df.sort_values(by=['srch_id']))
    return df_agg


def rank_variable_copy(df, variable):
    '''ranks on variable, returns copy, used for evaluate_score()'''
    df_copy = df.copy()
    # loop over search ids
    for search_id in search_ids:
        df_copy.loc[df_copy["srch_id"] == search_id] = df_copy[df_copy["srch_id"] == search_id].sort_values(variable, ascending=False).values 
    return df_copy

def pandas_to_csv(df, name):
    '''returns csv file with name out.csv from dataset'''
    df[["srch_id", "prop_id"]].to_csv(name, index=False)

def add_scores(df):
    '''add score column to dataframe'''
        def add_(row)
        '''adds scores to dataframe to evaluate performance'''
        val = 0
        if row["booking_bool"] == 1: 
            val += 5
        if row["click_bool"] == 1: 
            val += 1
        return val
    df["score"] = df.apply (lambda row: add_(row), axis=1) 
    return df


def evaluate_score(df):
    '''calculate the ndcg over all entries and averages, input dataframe'''
    score = 0.
    search_ids = df.srch_id.unique() # get unique id's
    #search_ids = df.srch_id.unique() # get unique id's
    y_true = rank_variable_copy(df, "scores")  # rank df on scores
    # loop over search ids
    for search_id in search_ids:
        y_true_id = y_true.loc[y_true["srch_id"] == search_id]["scores"].to_numpy() # get the true rank for search query as numpy
        y_score_id = df.loc[df["srch_id"] == search_id]["scores"].to_numpy() # get the output rank as numpy
        y_true_id = np.expand_dims(y_true_id, axis=0)# add dimension
        y_score_id = np.expand_dims(y_score_id, axis=0)
        print(y_true_id)
        print(y_score_id)
        score += ndcg_score(y_true_id, y_score_id) # calculate score
    
    score = score/len(search_ids)
    return "average NDCG:", score



