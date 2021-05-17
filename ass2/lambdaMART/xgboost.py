import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

def Groupshufflesplit(df, size=None, features):
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
    X_train = X_train[feature]
    X_val = X_val[features]

    return X_train, y_train, X_val, y_val, groups


# global variables
file_name = "xbg.pkl"

features = ['srch_id', 'site_id', 'visitor_location_country_id',
    'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
    'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
    'prop_location_score1', 'prop_location_score2',
    'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
    'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
    'srch_adults_count', 'srch_children_count', 'srch_room_count',
    'srch_saturday_night_bool', 'srch_query_affinity_score',
    'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv',
    'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
    'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
    'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
    'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
    'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
    'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
    'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
    'comp8_rate_percent_diff']

training = True

if __name__ == "main":

    # train the model
    if training == True:
        # import dataset
        df = pd.read_csv("training_set_VU_DM.csv")
        print("dataset loaded")

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
        #df["scores"] = df.apply (lambda row: add_scores(row), axis=1) 
        
        # extract relevant features from dataset

        X_train, y_train, X_val, y_val, groups = Groupshufflesplit(df, 100000, features) # change this value to get different sizes

        print("data prepared, starting training...")
        model.fit(X_train, y_train, group=groups, verbose=True) # train the model
        print("model trained")

        # validate the model
        y_val_predict = model.predict(X_val)

        # calculate ndcg
        val_ndcg = evaluate_score(y_val_predict, y_val, groups):
        print("validation_ndcg: ", val_ndcg)
        
        # save
        pickle.dump(model, open(file_name, "wb"))
        print("model saved")
        print("finished")

    # evaluate on testset
    if testing==True:

        # load model
        model = pickle.load(open(file_name, "rb"))
        print("xgboost loaded")

        # import dataset
        df_test = pd.read_csv("test_set_VU_DM.csv")
        print("dataset loaded")
        X_test = df_test[features]

        # predict
        y_test_predict = model.predict(X_test)
        print("predicted y values!")

        # create output df
        output = df_test[["srch_id", "prop_id"]]
        output["scores"] = y_test_predict

        # rank on score
        output_ranked = rank_variable(output, "scores")
        print("rearranged on scores per search query")

        # output to csv file
        pandas_to_csv(output_ranked, filename="out3.csv")
        print("saved as csv file")
        print("finished")