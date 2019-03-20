
#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./train.csv')
X = dataset.iloc[:, 1:785].values
y = dataset.iloc[:, 0].values

dataset1 = pd.read_csv('./test.csv')
#%%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p = 2)
classifier.fit(X, y)

#%%
import csv
with open('Submission.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    row1 = ['ImageId', 'Label']
    writer.writerow(row1)
    for i in range(1,28001,1):
        # print(i)
        X_test = dataset1.iloc[i - 1: i, :].values
        y_pred = classifier.predict(X_test)
        row = [i, y_pred[0]]
        writer.writerow(row)
        # print(y_pred)
#%%
csvFile.close()
