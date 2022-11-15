import pickle
from tensorflow import keras
import matplotlib.pyplot as plt

import utils
from scipy.io import loadmat


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


train = unpickle('./cifar-10-batches-py/data_batch_1')
test = unpickle('./cifar-10-batches-py/test_batch')
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

train_data = train[b'data'] / 255.0
train_data = train_data.reshape(len(train_data), 3, 32, 32).transpose(0, 2, 3, 1)
train_labels = train[b'labels']

test_data = test[b'data']
test_data = test_data.reshape(len(train_data), 3, 32, 32).transpose(0, 2, 3, 1)
test_labels = test[b'labels']

'''
plt.imshow(train_data[0])
plt.show()
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_data, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Tested accuracy: ", test_acc)
