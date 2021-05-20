# This file creates different datasets based on 

from utils_data import * 
import argparse
import pandas as pd 
import time

def process_dataset(dataset, imp, down, add_feats):
    """
    This funtion creates preprocessed datasets.  
    """
    t1 = time.time()
    
    df = dataset.copy()
    df = replace_comp_zero(df)
    if down: 
        df = downsample(df)
        print("succesfully downsampled in ", time.time() - t1, ' seconds')
        
    t2 = time.time()        
    
    if add_feats: 
        df = add_datetime(df)
        df = add_labels(df)
        df = price_order(df)
        df = count_window(df)
        print("succesfully added features", time.time() - t2, ' seconds')
        
    t3 = time.time()

    if imp == 'opp': 
        df = imputation_opposite(df)
        print("succesfully imputed in", time.time() - t3, ' seconds')
    if imp == 'min': 
        df = imputation_worst(df)
        print("succesfully imputed in", time.time() - t3, ' seconds')
    if imp == 'av': 
        df = imputation_average(df)
        print("succesfully imputed in", time.time() - t3, ' seconds')

    return df


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess the dataset.')

    # --------------------------------------------------------
    # # Imputation: opposite, worst, average case  
    # parser.add_argument('--imp', type=str, help='Options are: opp, min, av ', default='opp')
    # # Downsampling
    # parser.add_argument('--down', type=bool, help='Set to true if you want to downsample', default=False)
    # # Add features
    # parser.add_argument('--add_feats', type=bool, help='Set to true if you want to add features, false if not.', default=True)
    # # Normalize features
    # parser.add_argument('--norm', type=bool, help='Set to true if you want to normalize features, false if not.', default=False)
    # --------------------------------------------------------

    # Path to dataset, new datasets will also be saved here. 
    parser.add_argument('--folder', type=str, help='Set the path to data', default = 'data/')
    args = parser.parse_args()
    print("something is happening")
    print(args.folder+'training_set_VU_DM.csv')

    # read dataset 
    dataset = pd.read_csv(args.folder + 'training_set_VU_DM.csv')
    print("dataset has been read")

    imputations = ['opp', 'min', 'av']
    down_bools = [True, False]
    feat_bools = [True, False]

    for imp in imputations: 
        for down in down_bools: 
            if down: 
                str_d = "down_"
            else: 
                str_d = ""
            for feat in feat_bools: 
                if feat: 
                    str_f = "feat_"
                else: 
                    str_f = ""
                processed_df = process_dataset(dataset, imp, down, feat)
                df_name = 'proc_'+imp+'_'+str_d+str_f
                processed_df.to_csv(args.folder + df_name + '.csv')
                print('Successfully saved ', df_name, ' in folder', args.folder)
    
# Run this like: python3 create_datasets.py --folder ../data/