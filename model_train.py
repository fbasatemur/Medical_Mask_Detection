from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import numpy as np


def createCNNModel():

    model = Sequential()

    # layer 1
    model.add(Conv2D(200, (3,3), input_shape = data.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # layer 2
    model.add(Conv2D(100, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    # output layer
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy", 
                    optimizer = "adam",
                    metrics = ['accuracy'])

    return model

def showAccuracy(history):
      plt.plot(history.history["loss"], label = "training loss")
      plt.plot(history.history["val_loss"], label = "validation loss")
      plt.plot(history.history["accuracy"], label = "training accuracy")      
      plt.plot(history.history["val_accuracy"], label = "validation accuracy")
      plt.xlabel("epochs")
      plt.ylabel("values")
      plt.legend()
      plt.show()      
      
      
data = np.load("data_save.npy")
target = np.load("target_save.npy")

model = createCNNModel()

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1)

checkpoint = ModelCheckpoint("model-{epoch:03d}.model", monitor="val_loss", verbose=0, save_best_only=True, mode="auto")
history = model.fit(x_train, y_train, epochs=20, callbacks=[checkpoint], validation_split=0.2)

showAccuracy(history)

print(model.evaluate(x_test, y_test))