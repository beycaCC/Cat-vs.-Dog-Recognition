# Successed Version!

import os
import cv2
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

_DIR = "/Users/christinehe/Documents/CC_Projects/Internship/task3/dogscats/dogscats/fully_trainning_img"
IMG_NEW_SIZE=60
CATEGORIES = ['cats', 'dogs']


def pre_process_images():
    training_data = []
    broken_img_ct = 0
    for category in CATEGORIES:
        path = os.path.join(_DIR, category)
        for imgs in os.listdir(path):
            try:
                image=cv2.imread(os.path.join(path, imgs), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(image,(IMG_NEW_SIZE, IMG_NEW_SIZE))
                label = CATEGORIES.index(category)
                training_data.append([new_array, label])
            except Exception as e:
                # print("IMG is broken!")
                broken_img_ct += 1
                pass
    print('The number of broken imgs is ', broken_img_ct)
    return training_data


def build_model():
    # Build the model 
    model = Sequential([
        # 卷积层 Convolutional layer && 池化层 Pooling layer
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_NEW_SIZE, IMG_NEW_SIZE, 1), padding='same'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        # Conv2D(32, (3, 3), activation='relu'),
        # MaxPooling2D((2, 2)),
        # Conv2D(32, (3, 3), activation='relu'),
        # MaxPooling2D((2, 2)),
        # Conv2D(32, (3, 3), activation='relu'),
        # MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, input_shape=X.shape[1:], activation='relu'),
        # Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.summary()

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    model.fit(X, y, epochs=10, validation_split=0.1)
    
    return model


def process_each_img(path):
    test_img_path = path
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img, (IMG_NEW_SIZE, IMG_NEW_SIZE))
    new_array = np.array(new_array)
    new_array = new_array.reshape(-1, IMG_NEW_SIZE, IMG_NEW_SIZE, 1)
    return new_array


if __name__ == "__main__":
    processed_data = pre_process_images()
    print(len(processed_data))
    print(processed_data[0][1])

    random.shuffle(processed_data)

    X = []
    y = []
    for features, label in processed_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # print(X)
    # print(y)

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))

    X = X/255.0

    # print(X)

    X = X.reshape(-1, IMG_NEW_SIZE, IMG_NEW_SIZE, 1)
    

    TRAINED_MODEL = "dog_and_cat_trained.model"
    if(not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), TRAINED_MODEL))):
        print("Training Model...")
        trained_model = build_model()
        trained_model.save(TRAINED_MODEL)  # Store the current trained model
    else:
        print("Model training data found, skipping training...")


    # Start prediction
    print("Prediction starting...")
    loaded_model = keras.models.load_model(TRAINED_MODEL)
    prediction = loaded_model.predict([process_each_img('/Users/christinehe/Documents/CC_Projects/Internship/task3/cat_pic.jpeg')])
    print('Given Picture prediction is ', CATEGORIES[prediction.argmax()])


