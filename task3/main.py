from __future__ import absolute_import, division, print_function

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

import pandas as pd
import numpy as np

############################################################

num_output_values = 5

TRAIN_FILE = "train.h5"
TEST_FILE = "test.h5"
SAMPLE_FILE = "sample.csv"

TRAIN = pd.read_hdf(TRAIN_FILE)
x_data = TRAIN.loc[:, 'x1':'x100'].astype(float)
y_data = TRAIN.y.values
dummy_y = np_utils.to_categorical(y_data, 5)

TEST = pd.read_hdf(TEST_FILE)
test_id = range(45324, 45324 + len(TEST.index))
test_x_data = TEST.loc[:, 'x1':'x100'].values

############################################################

model = Sequential([
    Dense(32, input_shape=(100,)),
    Activation('relu'),
    Dense(num_output_values),
    Dropout(0.5),
    Activation('softmax'),
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_data, dummy_y, epochs=200, verbose=1)

prediction = model.predict(test_x_data)

############################################################

results = pd.DataFrame({'Id': test_id, 'y':  np.argmax(prediction, axis=1)})
results.to_csv(SAMPLE_FILE, index=False)
