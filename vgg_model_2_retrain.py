import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Define VGGNET model
def vgg_model():
    input_shape = (224, 224, 3)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv1',
                                 input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv2'))
    model.add(keras.layers.MaxPool2D(2))
    model.add(keras.layers.Dropout(.25))
    
    model.add(keras.layers.Conv2D(128, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv3'))
    model.add(keras.layers.Conv2D(128, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv4'))
    model.add(keras.layers.MaxPool2D(2))
    model.add(keras.layers.Dropout(.25))
    
    model.add(keras.layers.Conv2D(256, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv5'))
    model.add(keras.layers.Conv2D(256, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv6'))
    model.add(keras.layers.MaxPool2D(2))
    model.add(keras.layers.Dropout(.25))
    
    model.add(keras.layers.Conv2D(512, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv7'))
    model.add(keras.layers.Conv2D(512, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv8'))
    model.add(keras.layers.MaxPool2D(2))
    model.add(keras.layers.Dropout(.25))
    
    model.add(keras.layers.Conv2D(512, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv9'))
    model.add(keras.layers.Conv2D(512, (3, 3), strides=1, padding='valid',
                                 activation='relu', name='conv10'))
    model.add(keras.layers.MaxPool2D(2))
    model.add(keras.layers.Dropout(.25))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(2000, activation='relu'))
    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dense(1))
    
    adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(optimizer=adam, loss='mse')
    
    return model


print('Building training set...')
speeds = pd.read_csv('data/train.txt', names=['speed'])

y_full = speeds['speed'].values[1:]
X_full = np.load('new_processed_vid.npy')


print('Splitting training set...')
X_train, y_train = X_full[:1919], y_full[:1919]

X_test, y_test = X_full[1919:], y_full[1919:]


print('Loading model...')
model = keras.models.load_model('model.m')

print('Fitting model...')
model.fit(X_train, y_train, batch_size=126, epochs=5, validation_split=.1, verbose=1)

print('Evaluating model...')
print(model.evaluate(X_test, y_test, batch_size=126, verbose=1))

print('Saving model...')
keras.models.save_model(model, 'model.m')
print('Model saved.')
