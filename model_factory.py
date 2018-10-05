from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout

def create_baseline_model():
    model = Sequential()
    model.add(Flatten(input_shape=(96, 96, 1)))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(30))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(96, 96, 1), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dense(30))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
