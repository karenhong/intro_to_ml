from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn import utils
from sklearn import mixture

import pandas as pd
import numpy as np
import itertools as it

############################################################

num_output_values = 10
epochs = 350
pseudolabel_threshold = 0.9

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
lowest_bic = np.infty
bic = []
n_components = 10
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
    gmm.fit(x_unlabelled_data)
    bic.append(gmm.bic(x_unlabelled_data))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)

############################################################

# results = pd.DataFrame({'Id': test_id, 'y':  np.argmax(prediction, axis=1)})
# results.to_csv(SAMPLE_FILE, index=False)
