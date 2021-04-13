# Import 
import pandas as pd
import numpy as np 
import csv 
from datetime import datetime
import itertools
import seaborn as sn
import matplotlib.pyplot as plt

dataset = "dataframe_new.csv" 
df = pd.read_csv(dataset)
# df2 = df[df["t"]==3]

# corrMatrix = df2.corr()
# sn.heatmap(corrMatrix, annot=False)
# plt.show()

df_test = pd.read_csv("dataset_mood_smartphone.csv")

print(df_test[(df_test["variable"]=="appCat.builtin") & (df_test["value"] == 0)])