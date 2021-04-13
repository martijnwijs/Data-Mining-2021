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

dataset = "dataframe_normalized_outliers_removed.csv" 
df = pd.read_csv(dataset)

# aggregate the dataset 
df_agg = df.groupby(['no']).mean()
print(df_agg)

# TO DO: Create new df where it multiplies t=0 with 0.1, t=1 0.2, t=2 0.3, t=3 0.4 

# create numpy arrays with data
X = df_agg.iloc[:, 3:].to_numpy()
y = df_agg.iloc[:, 1].to_numpy()
print(y)
print(X)


# _ = plt.hist(y, bins='auto')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)

# train the SVM 

linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovr').fit(X_train, y_train)
poly = SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovr').fit(X_train, y_train)
sig = SVC(kernel='sigmoid', C=1, decision_function_shape='ovr').fit(X_train, y_train)

# # linear = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1)).fit(X_train, y_train)
# # rbf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovr')).fit(X_train, y_train)
# # poly = make_pipeline(StandardScaler(),SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovr')).fit(X_train, y_train)
# # sig = make_pipeline(StandardScaler(), SVC(kernel='sigmoid', C=1, decision_function_shape='ovr')).fit(X_train, y_train)

accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)

print("Accuracy Linear Kernel:", accuracy_lin)
print("Accuracy Polynomial Kernel:", accuracy_poly)
print("Accuracy Radial Basis Kernel:", accuracy_rbf)
print("Accuracy Sigmoid Kernel:", accuracy_sig)

# for clf in (linear, rbf, poly, sig): 
#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(acc) 