import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

image_directory='datasets/'

no_tumor_images=os.listdir(image_directory+ 'Training/no_tumor/')
meningioma_tumor_images=os.listdir(image_directory+ 'Training/meningioma_tumor/')
glioma_tumor_images=os.listdir(image_directory+ 'Training/glioma_tumor/')
pituitary_tumor_images=os.listdir(image_directory+ 'Training/pituitary_tumor/')
dataset=[]
label=[]

INPUT_SIZE=64

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(os.path.join(image_directory, 'Training' , 'no_tumor',image_name))
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(meningioma_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(os.path.join(image_directory,'Training', 'meningioma_tumor',image_name))
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in enumerate(glioma_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(os.path.join(image_directory,'Training', 'glioma_tumor',image_name))
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in enumerate(pituitary_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(os.path.join(image_directory,'Training', 'pituitary_tumor',image_name))
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(3)

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

y_train=to_categorical(y_train , num_classes=4)
y_test=to_categorical(y_test , num_classes=4)

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train,
    batch_size=16,
    verbose=1, epochs=50, 
    validation_data=(x_test, y_test),
    shuffle=False)

model.save('BrainTumorTypesImproved50EpochsCategorical.keras')

accuracy = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {accuracy}")