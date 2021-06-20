import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
import datetime


def preprocess_liver(df):
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    conditions = [df['Dataset'] == 1, df['Dataset'] == 2]
    values = [df.groupby('Dataset').mean()['Albumin_and_Globulin_Ratio'][1],
              df.groupby('Dataset').mean()['Albumin_and_Globulin_Ratio'][2]]
    df['Albumin_and_Globulin_Ratio'] = np.where(df['Albumin_and_Globulin_Ratio'].isnull(),
                                                np.select(conditions, values), df['Albumin_and_Globulin_Ratio'])
    X = df.values[:, :-1]
    X = MinMaxScaler().fit_transform(X)
    Y = df.values[:, -1] - 1
    return X, Y


def table_into_images(X):
    t = X.shape[1]
    sqr = int(np.round(np.sqrt(t)))
    images = np.zeros((X.shape[0], sqr + 1, sqr + 1))
    dif = (sqr + 1) ** 2 - t
    for i, image in enumerate(images):
        temp = np.ravel(image)
        temp[:-dif] = X[i].copy()
        images[i] = np.reshape(temp, (sqr + 1, sqr + 1))
    rgb_images = np.empty((X.shape[0], sqr+1, sqr+1, 3))
    for i, image in enumerate(images):
        rgb_images[i, :, :, 0] = image
        rgb_images[i, :, :, 1] = image
        rgb_images[i, :, :, 2] = image
    expanded_images = np.repeat(rgb_images, 20, axis=2)
    expanded_images = np.repeat(expanded_images, 20, axis=1)
    return expanded_images


if __name__ == '__main__':
    df = pd.read_csv('indian_liver_patient.csv')
    X, Y = preprocess_liver(df)
    X = table_into_images(X)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.shuffle(len(X_val)).batch(16)
    model = InceptionV3(include_top=False, weights='imagenet', input_shape=(80, 80, 3))
    for layer in model.layers:
        layer.trainable = False
    output = model.layers[-1].output
    output = Flatten()(output)
    output = Dense(64, activation='relu')(output)
    predictions = Dense(1, activation="sigmoid")(output)
    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    checkpoint_filepath = 'liver_image_net'
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[tensorboard_callback, checkpoint])
