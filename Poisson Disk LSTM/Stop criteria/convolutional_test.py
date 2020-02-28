'''Fonte do codigo original == "Building powerful image classification models using very little data"from blog.keras.io.```
data/
    train/
        class1/
            1.jpg
            2.jpg
            ...
        class2/
            1.jpg
            2.jpg
            ...
    validation/
        class1/
            1.jpg
            2.jpg
            ...
        class2/
            1.jpg
            2.jpg
            ...
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import cv2
import numpy as np

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = './train'
validation_data_dir = './validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 800
batch_size = 8

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.load_weights('stop_poisson-acc_0.897_err_0.2416.hdf5')
img = cv2.imread('1.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict(img)

print(classes)
img = cv2.imread('0.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict(img)

print(classes)
img = cv2.imread('2.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict(img)

print(classes)
img = cv2.imread('0-21.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict(img)

print(classes)
img = cv2.imread('0-22.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict(img)

print(classes)
img = cv2.imread('0-23.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict(img)

print(classes)