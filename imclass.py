import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define our example directories and files
base_dir ='C:\\Users\Stallone\Documents\PycharmProjects\example_code\ObjectCategories'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training pictures
train_scorpion_dir = os.path.join(train_dir, 'scorpion')
train_sea_horse_dir = os.path.join(train_dir, 'sea_horse')
train_soccer_ball_dir = os.path.join(train_dir, 'soccer_ball')
train_starfish_dir = os.path.join(train_dir, 'starfish')
train_stegosaurus_dir = os.path.join(train_dir, 'stegosaurus')
train_stop_sign_dir = os.path.join(train_dir, 'stop_sign')
train_sunflower_dir = os.path.join(train_dir, 'sunflower')

# Directory with our validation pictures
validation_scorpion_dir = os.path.join(validation_dir, 'scorpion')
validation_sea_horse_dir = os.path.join(validation_dir, 'sea_horse')
validation_soccer_ball_dir = os.path.join(validation_dir, 'soccer_ball')
validation_starfish_dir = os.path.join(validation_dir, 'starfish')
validation_stegosaurus_dir = os.path.join(validation_dir, 'stegosaurus')
validation_stop_sign_dir = os.path.join(validation_dir, 'stop_sign')
validation_sunflower_dir = os.path.join(validation_dir, 'sunflower')

train_scorpion_fnames = os.listdir(train_scorpion_dir)
train_sea_horse_fnames = os.listdir(train_sea_horse_dir)
train_soccer_ball_fnames = os.listdir(train_soccer_ball_dir)
train_starfish_fnames = os.listdir(train_starfish_dir)
train_stegosaurus_fnames = os.listdir(train_stegosaurus_dir)
train_stop_sign_fnames = os.listdir(train_stop_sign_dir)
train_sunflower_fnames = os.listdir(train_sunflower_dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # This is the source directory for training images
        target_size=(300, 200),  # All images will be resized to 150x150
        batch_size=5,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode=('categorical'))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 200),
        batch_size=5,
        class_mode=('categorical'))

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(300, 200, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(7, activation='sigmoid')(x)

# Uncomment this section to use, pre-trained Inception v3 model as classifier
# local_weights_file = 'C:\\Users\Stallone\Documents\PycharmProjects\example_code\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# pre_trained_model = InceptionV3(
#     input_shape=(300, 200, 3), include_top=False, weights=None)
# pre_trained_model.load_weights(local_weights_file)
#
# for layer in pre_trained_model.layers:
#   layer.trainable = False
#
# last_layer = pre_trained_model.get_layer('mixed7')
# print('last layer output shape:', last_layer.output_shape)
# last_output = last_layer.output
#
# # Flatten the output layer to 1 dimension
# x = layers.Flatten()(last_output)
# # Add a fully connected layer with 1,024 hidden units and ReLU activation
# x = layers.Dense(1024, activation='relu')(x)
# # Add a dropout rate of 0.2
# x = layers.Dropout(0.2)(x)
# # Add a final sigmoid layer for classification
# x = layers.Dense(7, activation='sigmoid')(x)

# # Configure and compile the model
# # model = Model(pre_trained_model.input, x)
# # model.compile(loss='categorical_crossentropy',
# #               optimizer=RMSprop(lr=0.0001),
# #               metrics=['acc'])

# Configure and compile the model
model = Model(img_input, output)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
       train_generator,
       steps_per_epoch=66,
       epochs=100,
       validation_data=validation_generator,
       validation_steps=33,
       verbose=2)


plt.interactive(False)
# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show(block=True)