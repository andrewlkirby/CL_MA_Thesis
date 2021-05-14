import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

seed = 333

y1 = [1] * 650
y0 = [0] * 350
y = y1 + y0

np.random.seed(seed)

val = 1000

x1 = (np.random.randint(1, 6, size = (val, 1)))
x2 = (np.random.randint(1, 6, size = (val, 1)))
x3 = (np.random.randint(1, 6, size = (val, 1)))
x4 = (np.random.randint(1, 6, size = (val, 1)))
x5 = (np.random.randint(1, 6, size = (val, 1)))

X = np.c_[x1, x2, x3, x4, x5]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

clf = LogisticRegression(penalty="l1", C=100, solver="liblinear").fit(X_train, y_train)
LRpreds = clf.predict(X_test)

#wrangling the pandas:
predsdf = pd.DataFrame(LRpreds)
predsdf = predsdf.rename(columns={0: 'LRpreds'})

actual = y_test
actualdf = pd.DataFrame(actual)

#determine accuracy:
predsdf['actual'] = actualdf
predsdf['correct'] = predsdf['LRpreds'] == (predsdf['actual'])
print(predsdf.head())
print("Finished!")
print("accuracy:", (predsdf['correct'].sum()) / len(predsdf['correct']))