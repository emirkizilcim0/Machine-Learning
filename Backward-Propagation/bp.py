import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hidden_layer_size = 32
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.summary()


learning_rate = 0.01
epochs = 1000

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(X_train)
        loss_value = loss_fn(y_train, logits)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value.numpy()}")