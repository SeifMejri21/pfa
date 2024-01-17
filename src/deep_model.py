import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class CNNModel:
    def __init__(self, input_shape, num_classes):
        self.model = None
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        self.model = Sequential()
        # Example architecture, modify as needed
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data, train_labels, batch_size=32, epochs=10, validation_data=None):
        if not self.model:
            raise Exception("Model not built. Call 'build_model' before 'train'.")
        self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data)

    def predict(self, image):
        if not self.model:
            raise Exception("Model not built. Call 'build_model' before 'predict'.")
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        pred = self.model.predict(image)
        pred = np.argmax(pred)
        return pred

    def optimize(self, learning_rate):
        if not self.model:
            raise Exception("Model not built. Call 'build_model' before 'optimize'.")
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def save_weights(self, file_path):
        self.model.save_weights(file_path)

    def load_weights(self, file_path):
        self.model.load_weights(file_path)

