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

dataset = "df_imp.csv" 
df = pd.read_csv(dataset)

df_feat = df
util_apps = ["appCat.builtin", 'appCat.finance', 'appCat.office', 'appCat.social', 
            'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
com_other_apps = ['appCat.communication', 'appCat.other']
entertain_apps = ['appCat.entertainment', 'appCat.game']

df_feat["util_apps"] = df[util_apps].sum(axis=1)
df_feat["com_other_apps"]=df[com_other_apps].sum(axis=1)
df_feat["entertain_apps"]=df[entertain_apps].sum(axis=1)

print(df_feat.columns)

df_feat = df_feat.drop(columns=["appCat.builtin", 'appCat.finance', 'appCat.office', 'appCat.social', 
                        'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather',
                        'appCat.communication', 'appCat.other', 'appCat.entertainment', 'appCat.game'])

df_feat.to_csv("df_attr_select.csv", index=False)
