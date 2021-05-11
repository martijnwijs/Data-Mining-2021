def replace_negative(df):
    """
    Replaces the missing variables .... with a negative value -10. 
    """
    df.orig_destination_distance.fillna(-10, inplace=True)
    df.visitor_hist_starrating.fillna(-10, inplace=True)
    df.visitor_hist_adr_usd.fillna(-10, inplace=True)
    df.prop_review_score.fillna(-10, inplace=True)
    df.prop_location_score2.fillna(-10, inplace=True)
    return df

def replace_zero(df):
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

def replace_affinity1(df):
    """
    Replaces the 
    """
    df.srch_query_affinity_score.fillna(1, inplace=True)
    #group used 1 because: as negative value imputation would have biased the results, as it was already present amongst the values
    return df

def replace_affinity2(df):
    df.srch_query_affinity_score.fillna(-350, inplace=True)
    #took a negative value lower than the minimum (which is -325), so still worst case scenario
    return df

