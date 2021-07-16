# encoding=utf-8

# Time : 2021/6/5 21:38 

# Author : 啵啵

# File : test_cnn.py 

# Software: PyCharm

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import tensorflow as tf


num_classes = 10
input_shape = (28,28,1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float16') / 255
x_test = x_test.astype('float16') / 255

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape,),
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes,activation="softmax"),
])

model.summary()
batch_size = 128
epochs=15
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.1)
scores = model.evaluate(x_test,y_test,verbose=0)
print(f"Test loss:{scores[0]}")
print(f"Test accuracy:{scores[1]}")



