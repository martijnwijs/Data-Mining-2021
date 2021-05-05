from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn import preprocessing


dataset = "df_imp.csv" 
df = pd.read_csv(dataset)

# aggregate the dataset 
df_agg = df[df.columns[3:]].multiply((df["t"]+1)*0.1, axis="index")
df_agg[["no", "target"]] = df[["no", "target"]]
df_agg = df_agg.groupby(['no', 'target'], as_index=False).sum()

print(df_agg)

df_X = df_agg[["mood", "activity"]]
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df_X)
scaled_df = pd.DataFrame(scaled_df, columns=df_X.columns)
X = scaled_df.to_numpy()

y = df_agg.iloc[:, 1].to_numpy()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = svm.SVC(kernel='linear')
clf = model.fit(X, y)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVM ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()