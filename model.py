# ASCII art synthesis using Model
# Modified from source: https://github.com/OsciiArt/Model

import json

import numpy as np
import tensorflow as tf

class Model:
    def __init__(self, weights_path="model/weight.hdf5"):
        self.weights_path = weights_path

        self.model = self.build_model()

        self.char_bitmap = self.load_bitmap()
        self.char_list = self.load_list()

    def build_model(self):
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(64, 64, 1)),

            # First 2D convolutional block
            tf.keras.layers.GaussianNoise(stddev=0.3),
            tf.keras.layers.Conv2D(
                16,
                (3, 3),
                padding="same",
                activation="linear",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.05
                ),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Second 2D convolutional block
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="linear",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.05
                ),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Third 2D convolutional block
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="linear",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.05
                ),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Fourth 2D convolutional block
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="linear",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.05
                ),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                411,
                activation="softmax",
                name="predictions",
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    mode="fan_avg", distribution="uniform"
                ),
            ),
        ])

        try:
            # Load the weights
            model.load_weights(self.weights_path)

            # Compile the model
            model.compile(
                loss="categorical_crossentropy", 
                optimizer="sgd", 
                metrics=["accuracy"]
            )

            print("Model loaded successfully!")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_list(self, path="model/output.json"):
        with open(path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        return data["character-list"]

    def load_bitmap(self, path="model/output.json"):
        with open(path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        char_bitmap = {}
        for char, bitmap in data['character-bitmap'].items():
            char_bitmap[char] = np.array(bitmap, dtype=bool)

        return char_bitmap

    def predict(self, img):
        if self.char_bitmap is None or self.char_list is None:
            raise ValueError("Character data not loaded. Call _load_character_data() first.")

        input_shape = [64, 64, 1]
        artwork = []

        for h in range((img.shape[0] - input_shape[0]) // 18):
            w = 0
            line = []
            while w <= img.shape[1] - input_shape[1]:
                patch = img[
                    h*18 : h*18+input_shape[0], 
                    w : w+input_shape[1]
                ]
                patch = patch.reshape([1, input_shape[0], input_shape[1], 1])

                predict = self.model(patch)
                predict = np.argmax(predict[0])
                char = self.char_list[predict]

                line.append(char)
                w += self.char_bitmap[char].shape[1]

            artwork.append(line)

        return artwork

    def generate(self, img):
        img = (img.astype(np.float32)) / 255
        predictions = self.predict(img)

        input_shape = [64, 64, 1]
        num_line = (img.shape[0] - input_shape[0]) // 18
        artwork = np.ones_like(img, dtype=np.uint8) * 255

        widths = []
        for h in range(num_line):
            w = 0
            for char in predictions[h]:
                char_width = self.char_bitmap[char].shape[1]
                patch = 255 - self.char_bitmap[char].astype(np.uint8) * 255
                artwork[h*18 : h*18+16, w : w+char_width] = patch
                w += char_width
            widths.append(w)

        artwork = artwork[0 : num_line*18+16, 0 : max(widths)]

        return (artwork, predictions)
