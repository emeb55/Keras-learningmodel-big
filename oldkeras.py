import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

training_data = pd.read_csv('/Users/emelyebarlow/Desktop/github-projects/Wine-AI-Keras-Eval-matrics/wine-3-1.csv')

training_y = training_data.pop('quality')
training_y.replace('good', 1, inplace=True)
training_y.replace('bad', 0, inplace=True)
training_x = training_data

arr_convert_x = training_x.to_numpy()
arr_convert_y = training_y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(arr_convert_x, arr_convert_y, test_size=0.4, random_state=2, shuffle=True)

from tensorflow.keras.callbacks import EarlyStopping

model = keras.Sequential([
    layers.InputLayer(shape=(11,)),
    layers.Dense(128, activation= 'relu'),
    layers.Dense(64, activation='relu'), #2 hidden layers seems to make model more accurate
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_training_history = model.fit(X_train, Y_train, epochs=60, validation_data=(X_test, Y_test))

# early_stop = EarlyStopping(
#     monitor='val_loss', 
#     patience=10,  # wait 10 epochs before stopping
#     restore_best_weights=True
# )

# history = model.fit(
#     training_x, training_y,
#     validation_data=(validation_X, validation_y),
#     epochs=100,
#     callbacks=[early_stop]
# )


plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(model_training_history.history['accuracy'], label='Training Accuracy')
ax1.plot(model_training_history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epochs')
ax1.legend()

ax2.plot(model_training_history.history['loss'], label='Training Loss')
ax2.plot(model_training_history.history['val_loss'], label='Validation Loss', linestyle='--')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epochs')
ax2.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
print("Training plots saved as training_metrics.png")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(test_loss, test_acc)

y_pred = model.predict(X_test)

print(confusion_matrix(Y_test, np.round(y_pred)))


tn, fp, fn, tp = confusion_matrix(Y_test, np.round(y_pred)).ravel()
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print(f"FN: {fn}")

from sklearn.metrics import ConfusionMatrixDisplay 

ConfusionMatrixDisplay.from_predictions(Y_test, np.round(y_pred))
y_pred = np.round(model.predict(X_test))
cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")




# import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from keras.datasets import reuters


# (x2_train, y2_train), (x2_test, y2_test) = reuters.load_data(num_words= 1000, test_split= 0.3, seed=42)

# print(f"X Train length: {len(x2_train)}")
# print(f"Y Train length: {len(y2_train)}")
# print(f"X Test length: {len(x2_test)}")
# print(f"Y Test length: {len(y2_test)}")

# print(f"Total data examples: {len(x2_train) + len(x2_test)}")


# print(x2_train[0])
# print(y2_train[0])

# vocabulary = reuters.get_word_index()
# vocabulary = {v:k for k,v in vocabulary.items()}

# for k,v in vocabulary.items():
#     print(k, v)

#     for word_idx in x2_train [0]:
#          print(vocabulary.get(word_idx - 3, '?'))

# import numpy as np 

# x2_train_enc = np.zeros( shape=(len(x2_train),10000))
# print(x2_train_enc.shape)

# for row_number, word_idx_seq in enumerate(x2_train):
#     x2_train_enc[ row_number, word_idx_seq] = 1

# print(x2_train_enc[0])

# x2_test_enc = np.zeros( shape=(len(x2_test), 10000))
# for row_number, word_idx_seq in enumerate(x2_test):
#     x2_test_enc[ row_number, word_idx_seq] = 1

# print(x2_test_enc[0])

# from keras.utils import to_categorical 
# y2_train_enc = to_categorical(y2_train)
# y2_test_enc = to_categorical(y2_test)

# print(y2_train_enc.shape)

# import tensorflow as tf
# from tensorflow import keras 
# from tensorflow.keras import layers

# model = keras.Sequential()
# model.add(layers.InputLayer(input_shape=(10000,)))
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(46, activation="softmax"))
# model.summary()


# from tensorflow.keras.optimizers import SGD

# model.compile(
#     loss='categorical_crossentropy', 
#     optimizer=SGD(learning_rate=0.05),
#     metrics=['accuracy']
# )

# x2_val = x2_train_enc[:1000]
# y2_val = y2_train_enc[:1000]

# x2_train_enc_rest = x2_train_enc[1000:]
# y2_train_enc_rest = y2_train_enc[1000:]

# model_training_history = model.fit(
#     x2_train_enc_rest,
#     y2_train_enc_rest,
#     batch_size = 512, 
#     epochs = 40, 
#     validation_data = (x2_val, y2_val)
# )

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

# loss= model_training_history.history['loss']
# val_loss = model_training_history.history['val_loss']

# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label="Validation Loss")

# plt.ylabel('Loss')
# plt.xlabel('Epochs')

# plt.axvline( x=60, color='pink')

# plt.legend()
# plt.show()

# print("Starting one-hot encoding of training set...")
# x2_train_enc = np.zeros(shape=(len(x2_train),10000))
# for row_number, word_idx_seq in enumerate(x2_train):
#     x2_train_enc[row_number, word_idx_seq] = 1
#     if row_number % 1000 == 0:
#         print(f"Processed {row_number} articles...")
# print("Finished one-hot encoding training set.")
