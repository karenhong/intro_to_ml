from sklearn import linear_model, metrics
from sklearn.model_selection import KFold

import numpy
import pandas as pd

TRAIN_FILE = "task1a/train.csv"
SAMPLE_FILE = "task1a/sample.csv"

# Import data from csv
dataframe = pd.read_csv(TRAIN_FILE)
x_data = dataframe.loc[:, 'x1':'x10'].values
y_data = dataframe.y.values

lambdas = [0.1, 1, 10, 100, 1000]
results = []

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
    avg = numpy.average(rmse)
    results.append(avg)

pd.DataFrame(results).to_csv(SAMPLE_FILE, index=False, header=False)
