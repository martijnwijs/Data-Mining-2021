import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import warnings
from evaluation import *
import numpy as np
from sklearn.metrics import ndcg_score
import pickle
from preprocessing.feature_engineering import *
from preprocessing.downsampling import *

warnings.filterwarnings(action='ignore', category=UserWarning)

def Groupshufflesplit(df, features, size=None, downsample_=False):
    '''splits the training dataset in a train + validation dataset, if argument size is given, only a part of the dataset is used to train faster
    groups give the indices of the different srch id's'''
    if size is not None: # use only part of dataset
        df = df[:size]

    gss = GroupShuffleSplit(test_size=.05, n_splits=1, random_state = 7).split(df, groups=df['srch_id'])

    X_train_inds, X_test_inds = next(gss)
    train_data= df.iloc[X_train_inds].copy()
    #print(train_data)
    if downsample_ == True: # downsampling
        train_data = downsample(train_data)
    train_data.sort_values('srch_id', inplace = True)

    X_train = train_data.loc[:, ~train_data.columns.isin(['scores'])]
    y_train = train_data.loc[:, train_data.columns.isin(['scores'])]
    groups_train = train_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    test_data= df.iloc[X_test_inds].copy()
    groups_test = test_data.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    #We need to keep the id for later predictions
    X_val = test_data.loc[:, ~test_data.columns.isin(['scores'])]
    y_val = test_data.loc[:, test_data.columns.isin(['scores'])]

    # remove features
    X_train = X_train[features]
    X_val = X_val[features]

    return X_train, y_train, X_val, y_val, groups_train, groups_test


# global variables


features = ['srch_id', 'site_id', 'visitor_location_country_id',
    'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
    'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
    'prop_location_score1', 'prop_location_score2',
    'prop_log_historical_price', 'price_usd', 'promotion_flag',
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
    'comp8_rate_percent_diff', 'year', 'month', 'hour']  ############# chagne this
'''
srch_id,date_time,site_id,visitor_location_country_id,visitor_hist_starrating,
visitor_hist_adr_usd,prop_country_id,prop_id,prop_starrating,prop_review_score,prop_brand_bool,
prop_location_score1,prop_location_score2,prop_log_historical_price,price_usd,promotion_flag,srch_destination_id,
srch_length_of_stay,srch_booking_window,srch_adults_count,srch_children_count,srch_room_count,srch_saturday_night_bool,
srch_query_affinity_score,orig_destination_distance,random_bool,comp1_rate,comp1_inv,comp1_rate_percent_diff,comp2_rate,comp2_inv,
comp2_rate_percent_diff,comp3_rate,comp3_inv,comp3_rate_percent_diff,comp4_rate,comp4_inv,comp4_rate_percent_diff,comp5_rate,comp5_inv,
comp5_rate_percent_diff,comp6_rate,comp6_inv,comp6_rate_percent_diff,comp7_rate,comp7_inv,comp7_rate_percent_diff,comp8_rate,comp8_inv,
comp8_rate_percent_diff,year,month,day,hour,minute
'''


if __name__ == "__main__":

    options = [ "Training", "Testing"]
    chosen_option = input("please chose between Training and Testing: ")

    # train the model
    if chosen_option == "Training":

        # import dataset
        name_dataset = input("give the name of the dataset + .csv: ")
        df = pd.read_csv(name_dataset)
        print("dataset loaded")
        print("size trainingset: ", len(df.index))
        # XBG initialization
        model = xgb.XGBRanker(  
            objective='rank:pairwise',
            silent = 1,
            random_state=42, 
            learning_rate=0.1,
            colsample_bytree=0.9, 
            eta=0.05, 
            max_depth=6,  # 6
            n_estimators=110,  #110
            subsample=0.75,
            scale_pos_weight = 1,  
            verbose_eval=True
            )  # scale pos class imbalance

        # add scores to dataframe
        #df["scores"] = df.apply (lambda row: add_scores(row), axis=1) # on for vu dm, off for rest ############
        
        # extract relevant features from dataset
        print("do you want to use the whole dataset? this means longer training time...")

        chosen_option = input("Yes, No: ")
        if chosen_option == "Yes":
            size = None
        else:
            size = 100000
        
        print("do you want to downsample...")

        chosen_option = input("Yes, No: ")
        if chosen_option == "Yes":
            downsample_ = True
        else:
            downsample_ = False
        
        X_train, y_train, X_val, y_val, groups_train, groups_test = Groupshufflesplit(df, features, size, downsample_)
        print("size trainingset after: ", len(X_train.index))
        print("size valset: ", len(X_val.index))
        print(y_val.head(10))
        print(X_train.head(10))
        #print("groups", groups_train)
        #print("data prepared, starting training...")
        model.fit(X_train, y_train, group=groups_train, verbose=True) # train the model
        #print("model trained")

        # validate the model
        y_val_predict = model.predict(X_val)

        #print("y_val_pred", y_val_predict)
        #print("y_Val", y_val.head(30))
        #print("x_Val", X_val.head(35))
        print("groups test", groups_test)

        '''
        # calculate ndcg
        val_ndcg = evaluate_score(y_val_predict, y_val, groups_test)
        print("validation_ndcg: ", val_ndcg)
        '''
        # debugging 
        # calculate ndcg
        X_val['predict'] = y_val_predict
        X_val['scores'] = y_val['scores'].values
        ndcg_df = X_val[['srch_id', 'predict', 'scores']]
        val_ndcg = evaluate_score_2(ndcg_df)
        print("validation_ndcg: ", val_ndcg)
        # save
        model_name = input("give the name of the model ")
        pickle.dump(model, open(model_name, "wb"))
        print("model saved")
        print("finished")

    # evaluate on testset
    if chosen_option == "Testing":

        # load model
        try:
            model_name = input("give the name of the model ")
            model = pickle.load(open(model_name, "rb"))
            print("xgboost loaded")
        except:
            print("no model with this name!")
            quit()

        # import dataset
        name_dataset = input("give the name of the dataset + .csv: ")
        
        df_test = pd.read_csv(name_dataset)
        #df_test = df_test[:1000] #####
        print("dataset loaded")
        X_test = df_test[features]
        print(df_test.shape[0])
        # predict
        y_test_predict = model.predict(X_test)
        print("predicted y values!")

        # create output df
        output = df_test[["srch_id", "prop_id"]]
        output["scores"] = y_test_predict
        print(output.shape[0])
        # rank on score
        output_ranked = rank_variable(output, "scores")
        print("rearranged on scores per search query")

        # output to csv file
        print(output_ranked.head(30))
        output_name = input("give the name of the output file: ")
        test_output_to_csv(output_ranked, output_name)
        print("saved as csv file")
        print("finished")