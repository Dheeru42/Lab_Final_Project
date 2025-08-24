# import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import os
warnings.filterwarnings("ignore")   # Hide all warnings
from tensorflow import keras 
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import Model,load_model
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from matplotlib import pyplot as plt
import seaborn as sns
import pathlib
import numpy as np

# batch size of image
batch_size = 16

# directory of images
data_dir = pathlib.Path("dataset")
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Number of images:", image_count)

# classe present in dataset
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

# Data Augumentation on image
#20% val 80% Train
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255, validation_split=0.2,
                                     rotation_range=20,
                                     zoom_range=0.15,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.15,
                                     horizontal_flip=True,)


# extract train image data from dataset directory
trainDataset = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                     classes=list(CLASS_NAMES),
                                                     target_size=(256, 256),
                                                     shuffle=True, subset="training")

# extract test image data from dataset directory
testDataset = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                    classes=list(CLASS_NAMES),
                                                    target_size=(256, 256),
                                                    shuffle=True, subset="validation")

# Design CNN Model 
model = Sequential()
model.add(Conv2D(16,kernel_size=(3,3), activation='relu',input_shape=(256,256,3))) # input layer
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3), activation='relu'))  # middle layer 1
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3), activation='relu')) # middle layer 2
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3), activation='relu')) # middle layer 3
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu')) # Dense layer 1
model.add(Dense(64, activation='relu')) # Dense layer 2
model.add(Dense(2, activation='softmax')) # output layer

# CNN Model Summary
model.summary()

# Compile Model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpoint = ModelCheckpoint("model.h5",
                             save_best_only=True,
                             verbose=1)

# train using the generators
history = model.fit(trainDataset,validation_data=testDataset,epochs=20, verbose=1, callbacks=[checkpoint])

# model evaluation
evaluation = model.evaluate(testDataset)
print("Val loss:", evaluation[0])
print("Val Accuracy:", evaluation[1]*100)

# Save model
model.save("model.h5")

# Model Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Model Loss Graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()