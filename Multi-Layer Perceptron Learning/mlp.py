import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Flatten, Dense
import keras

(X_train , y_train), (X_test, y_test) =  keras.datasets.mnist.load_data()

# Shuffling the data.
# 
# X_train_flattened = X_train.reshape(X_train.shape[0], -1)
# X_test_flattened = X_test.reshape(X_test.shape[0], -1)
# 
# train_df = pd.DataFrame(X_train_flattened)
# train_df['label'] = y_train
# 
# train_df = train_df.sample(frac=1).reset_index(drop=True)
# 
# X_train_shuffled = train_df.drop(columns='label').values
# y_train_shuffled = train_df['label'].values
# 

gray_scale = 255

print(type(X_train))
print(type(X_test))

X_train = X_train.astype('float32') / gray_scale
X_test = X_test.astype('float32') / gray_scale

print("Feature matrix (x_train):", X_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", X_test.shape)
print("Target matrix (y_test):", y_test.shape)

# Visualizing the data
fig, ax = plt.subplots(10 ,10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(X_train[k].reshape(28, 28), aspect='auto')
        k +=1
plt.show()


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=2000, validation_data=(X_test, y_test))

results = model.evaluate(X_test, y_test, verbose=0)
print("Test loss, Test accuracy: " , results)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title("Training and Validation Accuracy", fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Training and Validation Loss", fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)

plt.suptitle("Model Training Perfomance", fontsize=16)
plt.tight_layout()
plt.show()
