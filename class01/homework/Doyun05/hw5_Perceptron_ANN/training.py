#handwriten digit

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = tf.keras.models.load_model('mnist_ANN.h5')

mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()

num = 5
predict = model.predict(image_test[0:num])
print(predict)

print("* Prediction," ,np.argmax(predict, axis= 1))

plt.figure(figsize=(15, 15))
for idx in range(num) :
    sp = plt.subplot(1,5,idx+1) 
    plt.imshow(image_train[idx])
    plt.title(f'Label: {label_train[idx]}')

plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28)),  # 이미지 차원 조정
    Dense(128, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='softmax')
], name="Simple-ANN")

model.compile(
    optimizer = 'adam',
    loss='sparse_categorical_crossentropy', metrics = ['accuracy'],
)

model.fit(image_train, label_train, epochs=10, batch_size=10)
model.summary()
model.save("mnist_ANN.h5")

# draw test images with predicted value
NUM = 5
ptedict = model.predict(image_test[0:NUM])
print(predict)

print("* prediction,",np.argmax(predict, axis= 1))

plt.figure(fogsoze=(15,15))
for idx in range(NUM):
    sp = plt.subplot(1,5,idx+1)
    plt.inshow(image_test[idx])
plt.show()
