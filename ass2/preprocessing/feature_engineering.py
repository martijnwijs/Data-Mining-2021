import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

def preprocess_dates(df):
    df.date_time = pd.to_datetime(df.date_time)
    df['year'], df['month'], df['day'], df['hour'], df['minute'] = df.date_time.dt.year, df.date_time.dt.month, df.date_time.dt.day, df.date_time.dt.hour, df.date_time.dt.minute
    return df

def rank_variable(df, variable):
    '''ranks on variable, way faster now because of groupby'''
    df_agg = df.groupby("srch_id", group_keys=False)
    d = df_agg.apply(lambda x: x.sort_values(by='scores', ascending=False))
    d.reset_index()
    df_new = d[['srch_id','prop_id','scores']].reset_index()
    return df_new[['srch_id','prop_id','scores']]

def rank_variable_copy(df, variable):
    '''ranks on variable, returns copy, used for evaluate_score()'''
    df_copy = df.copy()
    # loop over search ids
    for search_id in search_ids:
        df_copy.loc[df_copy["srch_id"] == search_id] = df_copy[df_copy["srch_id"] == search_id].sort_values(variable, ascending=False).values 
    return df_copy

def test_output_to_csv(df, name):
    '''returns csv file with name from dataset'''
    df[["srch_id", "prop_id"]].to_csv(name, index=False)
    return

def add_scores(row):
    '''adds SCORES to dataframe'''
    val = 0
    if row["booking_bool"] == 1: 
        val += 5
    if row["click_bool"] == 1: 
        val += 1
    return val

'''
def evaluate_score(X_val, y_val):

    X_val_agg = X_val.groupby("srch_id")
    X_val_agg.apply(lambda _df: _df.sort_values(by=['srch_id']))

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
'''

def categorical_to_dummy(df, variable):
    '''transforms categorical variables into dummy variables'''
    dummies = pd.get_dummies(df[variable]).astype(np.int8)
    merged = pd.concat([df, dummies], axis='columns') # concatonate
    # drop original column
    # you have to drop one dummy variabale column, because of colliniearity
    final = merged.drop([variable, 1], axis='columns')
    return df


