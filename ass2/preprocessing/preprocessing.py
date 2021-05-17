# This file contains the main function for preprocessing 
from imputations import *
import pandas as pd 
import argparse


# def main(df, imp_neg, comp_zero):
#     """
#     Main function for processing  
#     """
    
#     if imp_neg: 
#         df = replace_negative(df) 
    
#     if comp_zero: 
#         df = replace_zero(df)


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Preprocess the dataset.')

#     # Imputation arguments 
#     parser.add_argument('--neg', type=bool, help='Set to true if you want to replace missing values for -10', default=False)
#     parser.add_argument('--comp_0', type=bool, help='Set to true if you want to replace missing competitor values with 0', default=False)

#     # Feature engineering arguments 
#     parser.add_argument('--date', type=bool, help='Set to true if you want to add date variables (month, year, day)', default=False)

#     args = parser.parse_args()

#     # read dataset 
#     df = pd.read_csv('./data/training_set_VU_DM.csv')

#     # process dataset 
#     processed_df = main(df, imp_neg=args.neg, comp_zero=args.comp_0)

#     # save dataset


#     print("The dataset was successfully preprocessed and saved.")

from feature_engineering import * 

if __name__ == "__main__":

    #kwargs = get_kwargs()
    #print(kwargs)
    df = pd.read_csv("training_set_VU_DM.csv")
    df_test = pd.read_csv("test_set_VU_DM.csv")

    # imputations
    df = replace_negative(df)
    df_test = replace_negative(df_test)

    df = replace_zero(df)
    df_test = replace_zero(df_test)

    df= replace_affinity1(df)  # 2 versions ? 
    df_test= replace_affinity1(df_test)

    # feature engineering
    df = preprocess_dates(df)
    df_test = preprocess_dates(df_test)

    # add scores to dataframe
    df["scores"] = df.apply (lambda row: add_scores(row), axis=1)
    df_test["scores"] = df_test.apply (lambda row: add_scores(row), axis=1)

    # categorical to dummy variables
    categorical = ['srch_id', 'site_id', 'visitor_location_country_id','prop_country_id','prop_id', 'srch_destination_id']
    for variable in categorical:
        df = categorical_to_dummy(df, variable)

    # save as csv
    # output to csv file
    output_name = input("give the name of the output file: ")
    pandas_to_csv(df, filename=output_name)
    print("finished")



