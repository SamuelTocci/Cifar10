import pickle
import tensorflow as tf  
from tensorflow import keras
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# other imports
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import GlobalMaxPooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model

# Display the version
print(tf.__version__)   
    
# Load in the data
cifar10 = tf.keras.datasets.cifar10
 
# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
 
# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()


# visualize data by plotting images
# fig, ax = plt.subplots(5, 5)
# k = 0
 
# for i in range(5):
#     for j in range(5):
#         ax[i][j].imshow(x_train[k], aspect='auto')
#         k += 1
 
# plt.show()

""""
The code below uses a convolutional neural network to train the model
"""


# Hyper Parameters - we can tweak these parameters to change the effectiveness of the model
learning_rate = 0.005 #this affects val_accuracy
dropout_1 = 0.1 # seems to affect the improvement of accurary per iteration.
dropout_2 = 0.1
batch_size = 32 #a smaller batch size means more batches to be processerd per epoch.

model = models.Sequential()
model.add(layers.Conv2D(32,(2,2), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(dropout_1))

model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(dropout_2))

model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
# Compile
model.compile(optimizer=optimizer,
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

epochs = 5
history = model.fit(x_train, 
    y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(x_test, y_test))

# Plot accuracy per iteration
# plt.plot(history.history['accuracy'], label='acc', color='red')
# plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
# plt.plot(history.history['val_loss'], label='val_los', color='blue')
# plt.legend()
# plt.show()



# number of classes
# K = len(set(y_train))

# calculate total number of classes
# for output layer
# print("number of classes:", K)

# Build the model using the functional API
# input layer
# i = Input(shape=x_train[0].shape)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
# x = BatchNormalization()(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

# x = Flatten()(x)
# x = Dropout(0.2)(x)

# # Hidden layer
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.2)(x)

# # last hidden layer i.e.. output layer
# x = Dense(K, activation='softmax')(x)

# model = Model(i, x)

# # model description
# model.summary()

# Fit
# r = model.fit(
# x_train, y_train, validation_data=(x_test, y_test), epochs=5)


# Fit with data augmentation
# Note: if you run this AFTER calling
# the previous model.fit()
# it will CONTINUE training where it left off
# batch_size = 32
# data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#   width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
 
# train_generator = data_generator.flow(x_train, y_train, batch_size)
# steps_per_epoch = x_train.shape[0] // batch_size
 
# r = model.fit(train_generator, validation_data=(x_test, y_test),
#               steps_per_epoch=steps_per_epoch, epochs=5)





# label mapping

labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

# select the image from our test dataset
image_number = 0

# display the image
plt.imshow(x_test[image_number])
plt.show()

# load the image in an array
n = np.array(x_test[image_number])

# reshape it
p = n.reshape(1, 32, 32, 3)

# pass in the network for prediction and
# save the predicted label
predicted_label = labels[model.predict(p).argmax()]

# load the original label
original_label = labels[y_test[image_number]]

# display the result
print("Original label is {} and predicted label is {}".format(
	original_label, predicted_label))

