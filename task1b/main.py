from sklearn import linear_model

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

pd.DataFrame(results).to_csv(SAMPLE_FILE, index=False, header=False)
