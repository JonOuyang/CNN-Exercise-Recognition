import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.metrics import confusion_matrix
import math

#Models
"""
A total of 15 models are created for testing.
5 models: 2, 3, 4, 5, 6 layers
3 different types of batch size: 32, 64, 128
5 model x 3 input = 15 models
"""

#6 CNN layers, 8 total layers
def build_model6(pretrained=None):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(), layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation="softmax"),
    ])
  
    if pretrained:
        model.load_weights(pretrained)
    return model
build_model6()

#5 CNN layers, 7 total layers
def build_model5(pretrained=None):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(), layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation="softmax"),
    ])

    if pretrained:
        model.load_weights(pretrained)
    return model
build_model5()

#4 CNN layers, 6 total layers
def build_model4(pretrained=None):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(), layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation="softmax"),
    ])

    if pretrained:
        model.load_weights(pretrained)
    return model
build_model4()

#3 CNN layers, 5 total layers
def build_model3(pretrained=None):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(), layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation="softmax"),
    ])

    if pretrained:
        model.load_weights(pretrained)
    return model
build_model3()

#2 CNN layers, 4 total layers
def build_model2(pretrained=None):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(), layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    #model.summary()
    
    if pretrained:
        model.load_weights(pretrained)
    return model
build_model2()

def train(model, sp, epochs=10):
    batch_size = 128
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))
    model.fit(
        x_train, 
        y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split = 0.1)
    model.evaluate(x_test, y2_test, verbose=1)
    model.save(sp)
    print("-=- Model Saved -=-")
    predicted_labels=np.argmax(model.predict(x_test), axis=1)
    confusion_mat = confusion_matrix(y_test, predicted_labels)
    print(confusion_mat)
    print(f'Model Accuracy based on testing data: {(np.trace(confusion_mat))/np.sum(confusion_mat)}')

#Model Training, all files are saved as h5 file types
#Testing has shown that accuracy peaks at approximately 50 epochs
"""
train(build_model5(), "model6(128).h5", epochs=10)
train(build_model5(), "model5(128).h5", epochs=10)
train(build_model4(), "model4(128).h5", epochs=10)
train(build_model3(), "model3(128).h5", epochs=10)
train(build_model2(), "model2(128).h5", epochs=10)
"""

#model.summary()
#model.predict(x_utest)
#model.evaluate(x_test, y2_test, verbose=1)
#print(confusion_mat)
