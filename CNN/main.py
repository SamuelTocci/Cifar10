import ssl
from os import path

import tensorflow as tf
from tensorflow import keras

ssl._create_default_https_context = ssl._create_unverified_context

# other imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

# Display the version
print(tf.__version__)

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
rebuild = True

#split the data
x_training, x_cv, y_training, y_cv = train_test_split(x_train, y_train, test_size=0.28, random_state=1000)
#to validate split is identical.
#x1, x2, y1, y2 = train_test_split(x_train, y_train, test_size=0.28, random_state=1000)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

""""
The code below uses a convolutional neural network to train the model
"""


############# HYPERPARAMETERS FIXED ##############
dropout_1 = 0.1  
dropout_2 = 0.1
batch_size = 32  
##################################################

########### HYPERPARAMETERS FOR TUNING ###########
learning_rate = 0.0005 
epochs = 30
##################################################


if not path.exists("saved_model") or rebuild:
    model = models.Sequential([
        layers.Conv2D(32, (2, 2), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(dropout_1),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(dropout_1),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    # Compile
    model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test))
    model.save("./test_overfit")    #change model here



model = keras.models.load_model("test_overfit") #change model here

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Tested accuracy: ", test_acc)

plt.figure(figsize = (8,8))

plt.subplot(1,2,1)
plt.plot(range(epochs) , history.history["accuracy"] , "r" , label = "Training Accuracy")
plt.plot(range(epochs) , history.history["val_accuracy"] , "b" , label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(range(epochs) , history.history["loss"] , "r" , label = "Training Loss")
plt.plot(range(epochs) , history.history["val_loss"] , "b" , label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.title("Loss")

plt.show()

predictions  = model.predict(x_test)

predictions_for_cm = predictions.argmax(1)

class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

cm = confusion_matrix(y_test,predictions_for_cm)
plt.figure(figsize=(10,8))
conf_mat = sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels = class_names)
conf_mat.set(ylabel="True Label", xlabel="Predicted Label")
plt.show();

# The code below is to look further into where the code went wrong

print(metrics.classification_report(y_test,predictions_for_cm))