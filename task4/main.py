from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import itertools as it

############################################################

num_output_values = 10

TRAIN_FILE = "train_labeled.h5"
UNLABELLED_TRAIN_FILE = "train_unlabeled.h5"
TEST_FILE = "test.h5"
SAMPLE_FILE = "sample_90.csv"

TRAIN = pd.read_hdf(TRAIN_FILE)
x_data = TRAIN.loc[:, 'x1':'x128'].astype(float)
y_data = TRAIN.y.values
dummy_y = np_utils.to_categorical(y_data, 10)

UNLABELLED_TRAIN = pd.read_hdf(TRAIN_FILE)
x_unlabelled_data = TRAIN.loc[:, 'x1':'x128'].astype(float)

TEST = pd.read_hdf(TEST_FILE)
test_id = range(30000, 38000)
test_x_data = TEST.loc[:, 'x1':'x128'].values

############################################################

# Feature Pre-processing
pca = PCA()
x_unlabelled_data = pca.fit_transform(x_unlabelled_data)
x_data = pca.transform(x_data)
test_x_data = pca.transform(test_x_data)

model = Sequential([
    Dense(32, input_shape=(128,), activation='relu'),
    Dense(20, init='uniform', activation='sigmoid'),
    Dense(num_output_values),
    Dropout(0.5),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_data, dummy_y, epochs=500, verbose=1)
combined_y_data = y_data
combined_x_data = x_data
i = 0
while i < 20:
    pred = model.predict(x_unlabelled_data)
    selectors = [max(x) > 0.90 for x in pred]
    y_unlabelled_data = list(it.compress(np.argmax(pred, axis=1), selectors))
    dummy_unlabelled_y = np_utils.to_categorical(y_unlabelled_data, 10)
    x_filtered_unlabelled_data = list(it.compress(x_unlabelled_data, selectors))
    print("Number of pseudolabels used: " + str(len(x_filtered_unlabelled_data)))

    if len(x_filtered_unlabelled_data) != 0:
        print("Here " + str(i))
        combined_x_data = np.append(x_data, x_filtered_unlabelled_data, axis=0)
        combined_y_data = np.append(dummy_y, dummy_unlabelled_y, axis=0)
    model.fit(combined_x_data, combined_y_data, epochs=500, verbose=1)
    i = i + 1

prediction = model.predict(test_x_data)

############################################################

results = pd.DataFrame({'Id': test_id, 'y':  np.argmax(prediction, axis=1)})
results.to_csv(SAMPLE_FILE, index=False)
