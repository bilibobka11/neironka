import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2

DS_TRAIN_MAN_DIR = '/train/men/'
DS_TRAIN_WOMEN_DIR = '/train/women/'
DS_TEST_MEN_DIR = '/test/men/'
DS_TEST_WOMEN_DIR = '/test/women/'
WORK_SIZE = (200, 200)
NUM_CLASSES = 2


def create_conv_model():
    input_shape = (WORK_SIZE[0], WORK_SIZE[1], 1)
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    model.summary()
    return model


def load_dataset(ds_dir):
    x_train, y_train = load_ds_train(ds_dir)
    x_test, y_test = load_ds_test(ds_dir)
    return x_train, y_train, x_test, y_test


def load_ds_train(ds_dir):
    x_train, y_train = [], []
    x_train, y_train = load_ds_image(x_train, y_train, os.path.join(ds_dir, DS_TRAIN_MAN_DIR[1:]), 1)
    x_train, y_train = load_ds_image(x_train, y_train, os.path.join(ds_dir, DS_TRAIN_WOMEN_DIR[1:]), 0)
    return x_train, y_train


def load_ds_test(ds_dir):
    x_test, y_test = [], []
    x_test, y_test = load_ds_image(x_test, y_test, os.path.join(ds_dir, DS_TEST_MEN_DIR[1:]), 1)
    x_test, y_test = load_ds_image(x_test, y_test, os.path.join(ds_dir, DS_TEST_WOMEN_DIR[1:]), 0)
    return x_test, y_test


def load_ds_image(x, y, dir_path, label):
    if not os.path.exists(dir_path):
        print(f"Папка не найдена: {dir_path}")
        return x, y
    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        img = load_img_from_file(img_path, WORK_SIZE)
        if img is not None:
            x.append(img)
            y.append(label)
    return x, y


def load_img_from_file(fname, imgsize=WORK_SIZE):
    img = cv2.imread(fname)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, imgsize)
    img = np.expand_dims(img, axis=2)
    return img


def learn_conv_model(model):
    ds_dir = "dataset"
    x_train, y_train, x_test, y_test = load_dataset(ds_dir)

    x_train = [x for x in x_train if x is not None]
    x_test = [x for x in x_test if x is not None]

    if not x_train or not x_test:
        raise ValueError("Нет загруженных изображений для обучения или теста")

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    x_train /= 255.0
    x_test /= 255.0

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    model.fit(
        datagen.flow(x_train, y_train, batch_size=16),
        steps_per_epoch=len(x_train) // 16,
        epochs=50,
        validation_data=(x_test, y_test),
        verbose=1
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Потери на тесте:', score[0])
    print('Точность на тесте:', score[1])
    print("Baseline Error: %.2f%%" % (100 - score[1] * 100))

    model.save('testIT3.keras')
    print("Модель сохранена как testIT3.keras")


if __name__ == '__main__':
    # model = create_conv_model()
    # learn_conv_model(model)

    model = load_model('testIT3.keras')

    tdir = "test_man.jpg"
    # tdir = "test_woman.jpg"

    img = cv2.imread(tdir)
    if img is None:
        print(f"Ошибка загрузки тестового изображения: {tdir}")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, WORK_SIZE)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)
        img /= 255.0
        res = model.predict(img, verbose=0)
        if res[0][1] > res[0][0]:
            print("Мужчина с вероятностью", res[0][1])
        else:
            print("Женщина с вероятностью", res[0][0])
