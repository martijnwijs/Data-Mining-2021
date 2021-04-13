import pandas as pd
import numpy as np

df2 = pd.read_csv('dataframe_new.csv')

# for x in df2['appCat.builtin']:
#     if x < 0:
#         print(x)

def filter_inacc(df, category):
    df[(df[category] < 0)] = np.nan # for all values under 0
    df[category].fillna(0, inplace=True) # replace with 0
    return df

categories_zero = [ 'activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.other', 'appCat.social', 'appCat.unknown', 'appCat.utilities', 'screen', 'appCat.finance', 'appCat.office', 'appCat.travel', 'appCat.weather', 'appCat.game']
#
for category in categories_zero:
     df2 = filter_inacc(df2, category)

# df2 = filter_inacc(df2 , 'appCat.builtin')

# for x in df2['appCat.builtin']:
#     if x < 0:
#         print(x)

def filter_inacc2(df, category):
    df[(df[category] < -2)] = np.nan # for all values under -2
    df[(df[category] > 2)] = np.nan # for values above 2
    df[category].fillna(0, inplace=True) # replace with 0
    return df

categories_2 = ['circumplex.arousal', 'circumplex.valence']

for category in categories_2:
     df2 = filter_inacc2(df2, category)

print(df2['circumplex.arousal'].describe())