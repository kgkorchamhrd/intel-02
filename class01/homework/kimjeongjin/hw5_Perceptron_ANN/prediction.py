import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()

model = tf.keras.models.load_model('mnist_ANN.h5')


NUM = 5
predict = model.predict(image_test[10:10+NUM])
print(predict)

print(" * prediction,",
      np.argmax(predict, axis = 1))

plt.figure(figsize=(15,15))
for idx in range(NUM):
    sp = plt.subplot(1,5,idx+1)
    plt.imshow(image_test[10+idx])
plt.show()