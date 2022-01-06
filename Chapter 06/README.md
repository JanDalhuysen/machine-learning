# Tensorflow 2

### Source Code

```python
import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)

print(predictions[0])
```

### Results

```powershell
PS C:\Python\Jupyter Notebook> python tensorflow1.py

2022-01-02 13:19:29.487104:
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169]
retrieving CUDA diagnostic information for host: DESKTOP-JAN

2022-01-02 13:19:29.488252:
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176]
hostname: DESKTOP-JAN

2022-01-02 13:19:29.495632:
I tensorflow/core/platform/cpu_feature_guard.cc:151]
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)
To use the following CPU instructions in performance-critical operations:  AVX AVX2

To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

Epoch 1/5
1875/1875 [==============================] - 5s 2ms/step - loss: 4.0864
Epoch 2/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.6889
Epoch 3/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.5792
Epoch 4/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.5559
Epoch 5/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.5222
313/313 [==============================] - 1s 2ms/step - loss: 0.5392

[6.6973323e-14 1.0524663e-26 1.9499370e-19 1.8896951e-15 2.0069127e-20
 1.6978759e-02 5.0885494e-16 2.2091405e-02 2.8700588e-12 9.6092981e-01]

9
```
