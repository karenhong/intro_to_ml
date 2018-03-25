from sklearn import linear_model, metrics 
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd

TRAIN_FILE = "train.csv"
SAMPLE_FILE = "sample.csv"

# Import data from csv
dataframe = pd.read_csv(TRAIN_FILE)
x_data = dataframe.loc[:, 'x1':'x5'].values
y_data = dataframe.y.values

linear = x_data
quadratic = np.square(x_data)
exponential = np.exp(x_data)
cosine = np.cos(x_data)
constant = np.ones_like(y_data).reshape(-1, 1)

features = np.hstack([linear, quadratic, exponential, cosine, constant])
results = []

regr = linear_model.LinearRegression()
regr.fit(features, y_data)
results.extend(regr.coef_)

# maybe try 
# 1. cross validation and/or 2. ridge regression 

x_data = features.copy()
lambdas = [0.1, 1, 10, 100, 1000]
lambdas.extend([i for i in range (10,100)])
results_error = []
results_coef = []

k = 10
kf = KFold(n_splits=k)

for l in lambdas:
    reg = linear_model.Ridge(alpha=l)
    rmse = []
    for train_index, test_index in kf.split(x_data):
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        reg.fit(X_train, y_train)
        prediction = reg.predict(X_test)
        rmse.append(metrics.mean_squared_error(y_test, prediction)**0.5)
    avg = np.average(rmse)
    results_error.append(avg)
    results_coef.append(reg.coef_)

min_index = results_error.index(min(results_error))
min_error = results_error[min_index]
min_result_coef = results_coef[min_index]

print ("RESULT")
print("- lambda ", lambdas[min_index])
print (min_error)
print (min_result_coef)


# ---- results with k cross
pd.DataFrame(min_result_coef).to_csv(SAMPLE_FILE, index=False, header=False)

