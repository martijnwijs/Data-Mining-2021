# helper functions for create_datasets 
import pandas as pd 
import numpy as np 

################################## ADDED FEATURES ########################################

def add_datetime(df):
    """
    Splits up date_time column into year month day hour
    """
    df.date_time = pd.to_datetime(df.date_time)
    df['year'], df['month'], df['day'], df['hour'] = df.date_time.dt.year, df.date_time.dt.month, df.date_time.dt.day, df.date_time.dt.hour
    df.drop(['date_time'], axis=1, inplace=True)
    return df

def labels(row):
    '''helper func for labels to dataframe'''
    if row["booking_bool"] == 1: 
        return 5
    elif row["click_bool"] == 1: 
        return 1
    else: 
        return 0
    
def add_labels(df): 
    df['label'] = df.apply(labels, axis=1)
    return df   


def price_order(df):
    df['price_order'] = df.groupby(['srch_id'])['price_usd'].apply(lambda x: np.argsort(x))
    return df

def count_window(df):
    """
    From paper "Combination of Diverse Ranking Models for Personalized Expedia Hotel Searches"
    count_window is a combination of: srch room count and srch booking window
    """
    df['count_window'] = df['srch_room_count'] * max(df["srch_booking_window"]) + df["srch_booking_window"] 
    return df 

def add_features(df): 
    df.date_time = pd.to_datetime(df.date_time)
    df['year'], df['month'], df['day'], df['hour'] = df.date_time.dt.year, df.date_time.dt.month, df.date_time.dt.day, df.date_time.dt.hour
    df.drop(['date_time'], axis=1, inplace=True)
    
    df['label'] = df.apply(labels, axis=1)
    df.sort_values(by=['srch_id', 'price_usd'], inplace=True)
    df['price_order'] = df.groupby(['srch_id'])['price_usd'].apply(lambda x: np.argsort(x))
    df = df.apply(lambda x: x + 1 if x.name == 'price_order' else x)





################################## DOWNSAMPLING ########################################

def downsample(df, k=1):
    """
    k determines how many negative points you want (click_bool = 0) for each positive point. Default is 1. 
    """
    df_majority = df[df.click_bool == 0]
    df_minority = df[df.click_bool == 1]
    
    sampled = df_majority.groupby("srch_id").sample(k) 
    
    df_downsampled = pd.concat([sampled, df_minority])
    df_downsampled.sort_values(by=['srch_id'], inplace=True)
    return df_downsampled


# def downsample2(df, k): 
#     df_majority = df[df.click_bool == 0]
#     df_minority = df[df.click_bool == 1]

#     for 



################################## IMPUTATIONS ########################################

def imputation_opposite(df): 
    """
    Imputes with opposite case
    """
    df.orig_destination_distance.fillna(-10, inplace=True)
    df.visitor_hist_starrating.fillna(-10, inplace=True)
    df.visitor_hist_adr_usd.fillna(-10, inplace=True)
    df.prop_review_score.fillna(-10, inplace=True)
    df.prop_location_score2.fillna(-10, inplace=True)
    df.srch_query_affinity_score.fillna(10, inplace=True)
    df.gross_bookings_usd.fillna(-10, inplace=True)
    return df 

def imputation_worst(df): 
    """
    Imputes with worst/minimum case
    """
    df.orig_destination_distance.fillna(df.orig_destination_distance.min(), inplace=True)
    df.visitor_hist_starrating.fillna(df.visitor_hist_starrating.min(), inplace=True)
    df.visitor_hist_adr_usd.fillna(df.visitor_hist_adr_usd.min(), inplace=True)
    df.prop_review_score.fillna(df.prop_review_score.min(), inplace=True)
    df.prop_location_score2.fillna(df.prop_location_score2.min(), inplace=True)
    df.srch_query_affinity_score.fillna(df.srch_query_affinity_score.min(), inplace=True)
    df.gross_bookings_usd.fillna(df.gross_bookings_usd.min(), inplace=True)
    return df 

def imputation_average(df): 
    """
    Imputes with average case
    """
    df.orig_destination_distance.fillna(df.orig_destination_distance.mean(), inplace=True)
    df.visitor_hist_starrating.fillna(df.visitor_hist_starrating.mean(), inplace=True)
    df.visitor_hist_adr_usd.fillna(df.visitor_hist_adr_usd.mean(), inplace=True)
    df.prop_review_score.fillna(df.prop_review_score.mean(), inplace=True)
    df.prop_location_score2.fillna(df.prop_location_score2.mean(), inplace=True)
    df.srch_query_affinity_score.fillna(df.srch_query_affinity_score.mean(), inplace=True)
    df.gross_bookings_usd.fillna(df.gross_bookings_usd.mean(), inplace=True)
    return df 

def replace_comp_zero(df):
    """ 
    Replaces the missing competitor variables values with 0. 
    """
    for i in range(1, 9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        df[rate].fillna(0, inplace=True)
        df[inv].fillna(0, inplace=True)
        df[diff].fillna(0, inplace=True)
    return df


################################## ... ########################################
