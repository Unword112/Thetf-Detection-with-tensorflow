import tensorflow as tf
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

train_data_dir = 'data/Frames/Anomaly'
test_data_dir = 'data/Frames/Normal'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/Frames/',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'data/Frames/',
    color_mode='grayscale', 
    target_size=(48, 48),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)), 
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  
])

print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
)

model.fit(train_generator,
          epochs=10,
          validation_data=test_generator
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.2f}")

model.save('model_anomaly_detection.h5')
model.save('saved_model/', save_format='tf')