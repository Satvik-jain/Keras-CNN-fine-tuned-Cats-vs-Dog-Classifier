import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Rescaling

image_height, image_width = 256, 256
batch_size = 32

from keras.applications.vgg16 import VGG16

conv_base = VGG16(
    weights = 'imagenet',
    include_top = False, # Not importing FC layers
    input_shape = (256, 256, 3)
)

train_ds = keras.utils.image_dataset_from_directory(
    directory = '/kaggle/input/dogs-vs-cats/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (image_height, image_width)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/kaggle/input/dogs-vs-cats/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (image_height, image_width)
)

conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential()

# Rescaling layer
model.add(Rescaling(1./255, input_shape=(image_height, image_width, 3)))

# Data Augmentation Layer
# model.add(data_augmentation)
model.add(RandomFlip("horizontal"))
model.add(RandomRotation(0.2))
model.add(RandomZoom(0.3))
model.add(RandomTranslation(height_factor=0, width_factor=0.2))
model.add(RandomContrast(0.2))

# VGG 16
model.add(conv_base)

# Fully Connected
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = keras.optimizers.RMSprop(learning_rate = 0.00001),
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])

epochs=10
model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=epochs
)

model.save("VGG16_fine_tunning_Data_Augmentation/model.keras")