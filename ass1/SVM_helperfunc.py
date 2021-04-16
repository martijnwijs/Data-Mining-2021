# Import 
import pandas as pd
import numpy as np 
import csv 
from datetime import datetime
import itertools
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from collections import Counter
from sklearn import tree

dataset = "dataframe_new.csv" 
df = pd.read_csv(dataset)

# aggregate the dataset 
df_agg = df.groupby(['no']).mean()

# create dataframe for features
df_X = df_agg.iloc[:, 2:]

# create numpy arrays with target values
y = df_agg.iloc[:, 0].to_numpy()

c = Counter()
c.update(y)
print(c)

weight = {k: len(y)/v for k, v in c.items()}
print(y)

# Get column names first
names = df_X.columns
print(names)

# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_X)
scaled_df = pd.DataFrame(scaled_df, columns=names)

X = scaled_df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)

# train the SVM 

dtree = tree.DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovr').fit(X_train, y_train)
poly = SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovr').fit(X_train, y_train)
sig = SVC(kernel='sigmoid', C=1, decision_function_shape='ovr').fit(X_train, y_train)

accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)
accuracy_dtree = dtree.score(X_test, y_test)

print("Accuracy Linear Kernel:", accuracy_lin)
print("Accuracy Polynomial Kernel:", accuracy_poly)
print("Accuracy Radial Basis Kernel:", accuracy_rbf)
print("Accuracy Sigmoid Kernel:", accuracy_sig)
print("Accuracy Decision Tree:", accuracy_dtree)