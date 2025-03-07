import tensorflow as tf
from keras.src.losses import sparse_categorical_crossentropy

# Convert to TensorFlow tensors
y_true = tf.constant([2])  # True label (integer)
y_pred = tf.constant([[0.1, 0.2, 0.7]])  # Predicted probabilities

# Compute loss
loss = sparse_categorical_crossentropy(y_true, y_pred)

# Convert tensor to NumPy for printing
print(loss.numpy())  