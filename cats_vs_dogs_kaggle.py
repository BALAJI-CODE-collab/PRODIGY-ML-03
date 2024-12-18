
import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Unzipping the dataset (if needed)
train_zip = "/kaggle/input/dogs-vs-cats/train.zip"
test_zip = "/kaggle/input/dogs-vs-cats/test1.zip"
train_dir = "/kaggle/working/train"
test_dir = "/kaggle/working/test"

if not os.path.exists(train_dir):
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")

if not os.path.exists(test_dir):
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")

# Setting up data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation',
)

# Building the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Compiling the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
)

# Preparing the test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory="/kaggle/working",
    classes=['test1'],
    target_size=(150, 150),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)

# Predicting on test data
predictions = model.predict(test_generator, steps=test_generator.samples)
labels = [1 if pred > 0.5 else 0 for pred in predictions]

# Saving the results
results = pd.DataFrame({
    'id': [int(filename.split('.')[0]) for filename in test_generator.filenames],
    'label': labels
})
results.sort_values(by='id', inplace=True)
results.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
