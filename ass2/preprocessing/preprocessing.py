# This file contains the main function for preprocessing 

from imputations import *
import pandas as pd 
import argparse


def main(df, imp_neg, comp_zero):
    """
    Main function for processing  
    """
    
    if imp_neg: 
        df = replace_negative(df) 
    
    if comp_zero: 
        df = replace_zero(df)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess the dataset.')

    # Imputation arguments 
    parser.add_argument('--neg', type=bool, help='Set to true if you want to replace missing values for -10', default=False)
    parser.add_argument('--comp_0', type=bool, help='Set to true if you want to replace missing competitor values with 0', default=False)

    # Feature engineering arguments 
    parser.add_argument('--date', type=bool, help='Set to true if you want to add date variables (month, year, day)', default=False)

    args = parser.parse_args()

    # read dataset 
    df = pd.read_csv('./data/training_set_VU_DM.csv')

    # process dataset 
    processed_df = main(df, imp_neg=args.neg, comp_zero=args.comp_0)

    # save dataset


    print("The dataset was successfully preprocessed and saved.")
