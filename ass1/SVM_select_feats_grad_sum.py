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
from sklearn.model_selection import cross_val_score

dataset = "df_attr_select.csv" 
df = pd.read_csv(dataset)

# aggregate the dataset where 3 days ago counts for 0.1, 2 days ago for 0.2, 1 day ago 0.3, now 0.4
df_agg = df[df.columns[3:]].multiply((df["t"]+1)*0.1, axis="index")
df_agg[["no", "target"]] = df[["no", "target"]]
df_agg = df_agg.groupby(['no', 'target'], as_index=False).sum()

# create numpy arrays with features and targets
df_X = df_agg.iloc[:, 2:]
y = df_agg.iloc[:, 1].to_numpy()

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df_X)
scaled_df = pd.DataFrame(scaled_df, columns=df_X.columns)

X = scaled_df.to_numpy()

# # train the SVM 
linear = SVC(kernel='linear', C=0.5, random_state=42)
rbf = SVC(kernel='rbf', gamma=1, C=0.5, random_state=42)
poly = SVC(kernel='poly', degree=3, C=0.5, random_state=42)
sig = SVC(kernel='sigmoid', C=0.5, random_state=42)

accuracy_lin = cross_val_score(linear, X, y, cv=10)
accuracy_poly = cross_val_score(poly, X, y, cv=10)
accuracy_rbf = cross_val_score(rbf, X, y, cv=10)
accuracy_sig = cross_val_score(sig, X, y, cv=10)

print("Accuracy Linear Kernel:", (accuracy_lin.mean(), accuracy_lin.std()))
print("Accuracy Polynomial Kernel:", (accuracy_poly.mean(), accuracy_poly.std()))
print("Accuracy Radial Basis Kernel:", (accuracy_rbf.mean(), accuracy_rbf.std()))
print("Accuracy Sigmoid Kernel:", (accuracy_sig.mean(),accuracy_sig.std()))


