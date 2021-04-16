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

# sn.heatmap(df[df.columns[4:-6]].corr("kendall"))
sn.heatmap(df.corr(method="kendall"))
plt.show()


# print(df_agg[df_agg.columns].corr()['target'])
# data[data.columns[1:]].corr()['special_col'][:-1]
