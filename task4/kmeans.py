import numpy as np

from sklearn.cluster import KMeans
from sklearn import datasets
from keras.utils import np_utils

import pandas as pd
import numpy as np

# np.random.seed(5)

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

num_output_values = 10

############################################################

TRAIN_FILE = "train_labeled.h5"
UNLABELLED_TRAIN_FILE = "train_unlabeled.h5"
TEST_FILE = "test.h5"
SAMPLE_FILE = "prediction.csv"

TRAIN = pd.read_hdf(TRAIN_FILE)
x_train = TRAIN.loc[:, 'x1':'x128'].astype(float)
y_data = TRAIN.y.values
y_train = np_utils.to_categorical(y_data, num_output_values)

UNLABELLED_TRAIN = pd.read_hdf(TRAIN_FILE)
x_unlabelled_data = TRAIN.loc[:, 'x1':'x128'].astype(float)

TEST = pd.read_hdf(TEST_FILE)
test_id = range(30000, 38000)
test_x_data = TEST.loc[:, 'x1':'x128'].astype(float)

############################################################

kmeans_model = KMeans(n_clusters=num_output_values, random_state=1).fit(x_unlabelled_data)
labels = kmeans_model.labels_
print (kmeans_model.labels_)
