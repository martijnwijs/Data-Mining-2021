import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

def Groupshufflesplit(df, size=None):
    '''splits the training dataset in a train + validation dataset, if argument size is given, only a part of the dataset is used to train faster
    groups give the indices of the different srch id's'''
    if size is not None: # use only part of dataset
        df = df[:size]

    gss = GroupShuffleSplit(test_size=.40, n_splits=1, random_state = 7).split(df, groups=df['srch_id'])

    X_train_inds, X_test_inds = next(gss)
    train_data= df.iloc[X_train_inds]
    X_train = train_data.loc[:, ~train_data.columns.isin(['srch_id','scores'])]
    y_train = train_data.loc[:, train_data.columns.isin(['scores'])]
    groups = train_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    test_data= df.iloc[X_test_inds]

    #We need to keep the id for later predictions
    X_val = test_data.loc[:, ~test_data.columns.isin(['scores'])]
    y_val = test_data.loc[:, test_data.columns.isin(['scores'])]


    # remove features
    return X_train, y_train, X_val, y_val, groups


if __name__ == "main":

    # import dataset
    df = pd.read_csv("training_set_VU_DM.csv")

    # XBG initialization
    model = xgb.XGBRanker(  
        objective='rank:pairwise',
        random_state=42, 
        learning_rate=0.1,
        colsample_bytree=0.9, 
        eta=0.05, 
        max_depth=6, 
        n_estimators=110, 
        subsample=0.75 
        )

    # add scores to dataframe
    df["scores"] = df.apply (lambda row: add_scores(row), axis=1) 
    
    X_train, y_train, X_val, y_val, groups = Groupshufflesplit(df, 100000) # change this value to get different sizes
    model.fit(X_train, y_train, group=groups, verbose=True) # train the model

    # validate the model
    y_val_model = model.predict(X_validate)

    #insert to X_val
    X_val["scores"] = y_val_model

    # calculate ndcg
    evaluate_score(X_val, y_val):

    # rank the testset
    #rank_variable(output, "scores")
