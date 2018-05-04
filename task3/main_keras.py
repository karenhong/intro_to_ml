'''
Other things to try:
- 1d convo? vs 2d
- build up cnn with more layers
'''


import numpy as np
import os
import tempfile
import pandas as pd
import keras
from keras import backend as K
from keras import layers
from keras.models import Sequential
from keras.datasets import mnist

import tensorflow as tf

TRAIN = pd.read_hdf("train.h5", "train")
TEST = pd.read_hdf("test.h5", "test")
features = TRAIN.loc[:, 'x1':'x100'].values
labels = TRAIN.y.values
test_features = TEST.loc[:, 'x1':'x100'].values


if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the TensorFlow backend,'
                       ' because it requires the Datset API, which is not'
                       ' supported on other platforms.')


def cnn_layers(inputs):
    print(inputs)
    x = layers.Conv2D(32, (2, 2),
                      activation='relu', input_shape=(100, 1, 1), padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return predictions

def reshapeToImages(vector):
  image_vector = []
  for row in vector:
    image_vector.append(np.reshape(row, (-1,10)))
  return image_vector

batch_size = 128
buffer_size = 10000
steps_per_epoch = int(np.ceil(60000 / float(batch_size)))  # = 469
epochs = 5
num_classes = 10

################################
######## LOAD DATA ########
################################

# array of 1x100 to array of 10x10, "small images"
x_train = reshapeToImages(features)
y_train = labels
x_test = None
y_test = None
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype(np.float32) / 255 # dont need to convert to float, value salready in float
x_train = np.expand_dims(x_train, -1)
y_train = tf.one_hot(y_train, num_classes)

# Create the dataset and its associated one-shot iterator.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size)
dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()


################################
######## BUILD & TRAIN MODEL ########
################################


# Model creation using tensors from the get_next() graph node.
inputs, targets = iterator.get_next()
model_input = layers.Input(tensor=inputs)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

# train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy'],
#                     target_tensors=[targets])
# train_model.summary()

# train_model.fit(epochs=epochs,
#                 steps_per_epoch=steps_per_epoch)

# Save the model weights.
weight_path = os.path.join(tempfile.gettempdir(), 'saved_wt.h5')
train_model.save_weights(weight_path)



################################
######## APPLY MODEL ON TEST SET ########
################################


# # Second session to test loading trained model without tensors.
x_test = reshapeToImages(test_features)
x_test = np.expand_dims(x_test, -1)

x_test_inp = layers.Input(shape=x_test.shape[1:])
test_out = cnn_layers(x_test_inp)
test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights(weight_path)
test_model.compile(optimizer='rmsprop',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

# loss, acc = test_model.evaluate(x_test, y_test, num_classes)
# print('\nTest accuracy: {0}'.format(acc))

# a = train_model.predict_classes(x_test, verbose=1) # only for sequential
y_prob = test_model.predict(x_test) 
y_classes = y_prob.argmax(axis=-1)
print(y_classes)
# print (a)
# y_prob = train_model.predict(test_features) 
# y_classes = y_prob.argmax(axis=-1)
# print (y_classes)

# Clean up the TF session.
K.clear_session()

################################
######## EXPORT RESULTS ########
################################
