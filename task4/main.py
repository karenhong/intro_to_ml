from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import utils
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import itertools as it

############################################################

num_output_values = 10
epochs = 1000
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

x_unlabelled_data = np.append(x_unlabelled_data, test_x_data, axis=0)
############################################################

# Pre-processing
# x_train = preprocessing.scale(x_train)
# x_unlabelled_data = preprocessing.scale(x_unlabelled_data)
# test_x_data = preprocessing.scale(test_x_data)

pca = PCA()
x_unlabelled_data = pca.fit_transform(x_unlabelled_data)
x_train = pca.transform(x_train)
test_x_data = pca.transform(test_x_data)

class_weights = utils.class_weight.compute_class_weight('balanced', np.unique(y_data), y_data)

model = Sequential([
    Dense(32, input_shape=(128,), activation='relu'),
    Dense(20, kernel_initializer='uniform', activation='sigmoid'),
    Dense(num_output_values),
    Dropout(0.5),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, verbose=1, class_weight=class_weights)

i = 0
while i < 3:

    pred = model.predict(x_unlabelled_data)

    selectors = [max(x) > pseudolabel_threshold for x in pred]
    y_unlabelled_data = list(it.compress(np.argmax(pred, axis=1), selectors))
    pseudolabelled_y = np_utils.to_categorical(y_unlabelled_data, num_output_values)
    pseudolabelled_x = list(it.compress(x_unlabelled_data, selectors))
    print("Number of pseudolabels used: " + str(len(pseudolabelled_x)))
    combined_y = np.append(y_data, y_unlabelled_data)
    class_weights = utils.class_weight.compute_class_weight('balanced', np.unique(combined_y), combined_y)

    if len(pseudolabelled_x) != 0:
        X = np.append(x_train, pseudolabelled_x, axis=0)
        y = np.append(y_train, pseudolabelled_y, axis=0)
        model.fit(X, y, epochs=epochs, verbose=1, class_weight=class_weights)
    i += 1

prediction = model.predict(test_x_data)

############################################################
results = pd.DataFrame({'Id': test_id, 'y':  np.argmax(prediction, axis=1)})
results.to_csv(SAMPLE_FILE, index=False)
