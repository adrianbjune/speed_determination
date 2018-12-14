import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import boto3


# METHODS ----------


# Returns two arrays, one containing a list of all the RGB value arrays of Farneback flow
# and the other containing the matching speeds
def build_training():
    
    cap = cv2.VideoCapture('data/train.mp4')
    
    speeds = pd.read_csv('data/train.txt', names=['speed'])
    
    ret, frame = cap.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    
    rgbs = []
    while ret:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        rgbs.append(cv2.resize(rgb, (224,224), interpolation=cv2.INTER_AREA))
        
        if len(rgbs)%100==0:
            print('{} frames done'.format(len(rgbs)))
    
    return np.array(rgbs), speeds.values[1:]

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
    
    

# RUNNING ------------

print('Building training set...')
X_full, y_full = build_training()

print('Splitting training set...')
X_train, y_train = X_full[:1919], y_full[:1919]

X_train, y_train = X_full[1919:], y_full[1919:]


print('Constructing model...')
model = vgg_model()

print('Fitting model...')
model.fit(X_train, y_train, batch_size=100, epochs=50, validation_split=.1, verbose=2)

print('Evaluating model...')
print(model.evaluate(X_test, y_test, batch_size=100, verbose=1))

print('Saving model...')
keras.models.save_model(model, 'model.m')
print('Model saved.')

