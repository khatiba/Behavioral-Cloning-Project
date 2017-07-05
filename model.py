# Copied from the Jupyter notebook
# Behavioral Cloning Project

### Load the CSV Data and Fix Paths

import math
import cv2
import csv
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D


samples = []

for datapath in ['./track-1', './track-2']:
    
    with open('./{}/driving_log.csv'.format(datapath)) as fp:
        reader = csv.reader(fp)
        
        adjustments = [0, 0.2, -0.2]
        
        for line in reader:
            for i in range(3):
                path = './{}/IMG/{}'.format(datapath, line[i].split('/')[-1])
                measurement = float(line[3]) + adjustments[i]
                
                samples.append((path, measurement))

samples = shuffle(samples)

num_samples = len(samples)
print('Total samples: {}'.format(len(samples)))

training_samples = samples[:int(num_samples * 0.75)]
num_training_samples = len(training_samples)
print('Training samples: {}'.format(num_training_samples))

validation_samples = samples[int(num_samples * 0.75):]
num_validation_samples = len(validation_samples)
print('Validation samples: {}'.format(num_validation_samples))

test_image = cv2.imread(samples[0][0])
input_shape = test_image.shape
print('Input shape: {}'.format(input_shape))

plt.imshow(test_image)
plt.show()

epochs = 10
batch_size = 512


def batch_generator(samples):
    """
    Takes in samples and chunks them and loads the image, does some
    preprocessing, duplicates and flips and returns the batch.
    """
    num_batches = int(math.ceil(len(samples) / batch_size))
    
    while True:
        
        for i in range(0, num_batches):
            
            start = i * batch_size
            end = start + batch_size
            batch = samples[start:end]
            
            batch_inputs = []
            batch_targets = []
            
            for path, measurement in batch:
                image = cv2.imread(path)
                image = cv2.GaussianBlur(image, (3, 3), 0)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                
                batch_inputs.append(image)
                batch_targets.append(measurement)
                
                batch_inputs.append(np.fliplr(image))
                batch_targets.append(-1.0 * measurement)
            
            yield (np.array(batch_inputs), np.array(batch_targets))


### Network Architecture


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(
        generator=batch_generator(training_samples),
        steps_per_epoch=math.ceil(len(training_samples) / batch_size),
        validation_data=batch_generator(validation_samples),
        validation_steps=math.ceil(len(validation_samples) / batch_size),
        epochs=10,
        initial_epoch=0,
        verbose=1
)


### plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


model.save('model.h5')

print('Model saved.')
