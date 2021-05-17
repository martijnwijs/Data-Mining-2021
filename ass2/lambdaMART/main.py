# lambdaRANK

import numpy as np
import pandas as pd 
from LambdaRankNN import LambdaRankNN

TRAIN_DATA = "../data/set_neg_zero_aff1.csv"

# Read the data into memory
training_data = pd.read_csv(TRAIN_DATA,nrows= 2500000)
validation_data = pd.read_csv(TRAIN_DATA,skiprows=2500000,nrows = 1000000, header=False, names= training_data.columns)
test_data = pd.read_csv(TRAIN_DATA,skiprows= 3500000,nrows = 1000000, header=False,names= training_data.columns)

col_names = list(training_data.columns)
col_names.remove('click_bool')
col_names.remove('booking_bool')
col_names.remove('srch_id')

# A relevance function to define the relevance score for NDCG
def relevance(a):
    if a[0] == a[1] == 1:
        return 5
    elif a[0] == 1 and a[1] == 0:
        return 1
    else:
        return 0

# Generate the SVMLight format file

X_train = training_data[col_names].values
y_train = training_data.iloc[:,-2:].apply(relevance,axis = 1)

validation_data[col_names].values,validation_data.iloc[:,-2:].apply(relevance,axis = 1)
col_names].values,test_data.iloc[:,-2:].apply(relevance,axis = 1)

dump_svmlight_file(training_data_new[col_names].values,training_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_training_avg_mean_std_competitors_m2.txt',query_id=training_data_new.srch_id)
dump_svmlight_file(validation_data_new[col_names].values,validation_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_validation_avg_mean_std_competitors_m2.txt',query_id=validation_data_new.srch_id)
dump_svmlight_file(test_data_new[col_names].values,test_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_test_avg_mean_std_competitors_m2.txt',query_id = test_data_new.srch_id)



# generate query data
X = np.array([[0.2, 0.3, 0.4],
              [0.1, 0.7, 0.4],
              [0.3, 0.4, 0.1],
              [0.8, 0.4, 0.3],
              [0.9, 0.35, 0.25]])
y = np.array([0, 1, 0, 0, 2])
qid = np.array([1, 1, 1, 2, 2])

# train model
ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(X, y, qid, epochs=5)
y_pred = ranker.predict(X)
ranker.evaluate(X, y, qid, eval_at=2)