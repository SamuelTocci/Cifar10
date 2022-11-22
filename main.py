import ssl
from os import path

import tensorflow as tf
from tensorflow import keras

ssl._create_default_https_context = ssl._create_unverified_context

# other imports
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models

# Display the version
print(tf.__version__)

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
epochs = 7
rebuild = True

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
learning_rate = 0.005  # this affects val_accuracy
dropout_1 = 0.1  # seems to affect the improvement of accuracy per iteration.
dropout_2 = 0.1
batch_size = 32  # a smaller batch size means more batches to be processed per epoch.

if not path.exists("./saved_model") or rebuild:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(dropout_1))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(dropout_2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    # Compile
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    model.save("./saved_model")

model = keras.models.load_model("./saved_model")

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Tested accuracy: ", test_acc)

# label mapping

labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

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
