import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Rescaling
from keras.layers import Rescaling, Resizing, RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast, Input

image_height, image_width = 256, 256
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    directory = '/kaggle/input/dogs-vs-cats/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
#     image_size = (image_height, image_width)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/kaggle/input/dogs-vs-cats/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
#     image_size = (image_height, image_width)
)

model = Sequential()

# Resizing and Rescalling
model.add(Input(shape=(image_height, image_width, 3)))
model.add(Resizing(image_width, image_height))
model.add(Rescaling(1./255))

# Data Augmentation Layer
# model.add(data_augmentation)
model.add(RandomFlip("horizontal"))
model.add(RandomRotation(0.2))
model.add(RandomZoom(0.3))
model.add(RandomTranslation(height_factor=0, width_factor=0.2))
model.add(RandomContrast(0.2))

model.add(Conv2D(32, kernel_size = (3,3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid', strides = 2))

model.add(Conv2D(64, kernel_size = (3,3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid', strides = 2))

model.add(Conv2D(128, kernel_size = (3,3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid', strides = 2))
# model.add(GlobalMaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

epochs=10
model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=epochs
)

model.save('Simple_CNN_Data_Augmentation/model.keras')