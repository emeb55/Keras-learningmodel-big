
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.datasets import reuters


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words= 1000, test_split= 0.3, seed=42)

# print(f"X Train length: {len(x_train)}")
# print(f"Y Train length: {len(y_train)}")
# print(f"X Test length: {len(x_test)}")
# print(f"Y Test length: {len(y_test)}")

# print(f"Total data examples: {len(x_train) + len(x_test)}")


# print(x_train[0])
# print(y_train[0])

vocabulary = reuters.get_word_index()


vocabulary = {v:k for k,v in vocabulary.items()}


# for k,v in vocabulary.items():
#     print(k, v)

#     for word_idx in x_train [0]:
#         print(vocabulary.get(word_idx - 3, '?'))

import numpy as np 

x_train_enc = np.zeros( shape=(len(x_train),10000))
print(x_train_enc.shape)

for row_number, word_idx_seq in enumerate(x_train):
    x_train_enc[ row_number, word_idx_seq] = 1

print(x_train_enc[0])

x_test_enc = np.zeros( shape=(len(x_test), 10000))
for row_number, word_idx_seq in enumerate(x_test):
    x_test_enc[ row_number, word_idx_seq] = 1

print(x_test_enc[0])

from keras.utils import to_categorical 
y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)

print(y_train_enc.shape)

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(10000,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))
model.summary()


from tensorflow.keras.optimizers import SGD

model.compile(
    loss='categorical_crossentropy', 
    optimizer=SGD(learning_rate=0.05),
    metrics=['accuracy']
)

x_val = x_train_enc[:1000]
y_val = y_train_enc[:1000]

x_train_enc_rest = x_train_enc[1000:]
y_train_enc_rest = y_train_enc[1000:]

model_training_history = model.fit(
    x_train_enc_rest,
    y_train_enc_rest,
    batch_size = 512, 
    epochs = 40, 
    validation_data = (x_val, y_val)
)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

loss= model_training_history.history['loss']
val_loss = model_training_history.history['val_loss']

plt.figure()
plt.plot(model_training_history.history['loss'], label="Training Loss")
plt.plot(model_training_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Baseline Model (ReLU)")
plt.legend()


pred_y = model.predict(x_test_enc)

print(pred_y[0].shape)
print(pred_y[0])
print(np.sum(pred_y[0]))

print(np.argmax(pred_y[0]))


leaky_model = keras.Sequential([
    layers.InputLayer(input_shape=(10000,)),
    layers.Dense(128, activation="leaky_relu"),  
    layers.Dense(64, activation="leaky_relu"),
    layers.Dense(46, activation="softmax")
])


from tensorflow.keras.optimizers import SGD

model.compile(
    loss='categorical_crossentropy', 
    optimizer=SGD(learning_rate=0.05),
    metrics=['accuracy']
)

x_val = x_train_enc[:1000]
y_val = y_train_enc[:1000]

x_train_enc_rest = x_train_enc[1000:]
y_train_enc_rest = y_train_enc[1000:]

model_training_history = model.fit(
    x_train_enc_rest,
    y_train_enc_rest,
    batch_size = 512, 
    epochs = 150, 
    validation_data = (x_val, y_val)
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.05), 
    metrics=['accuracy']
)


import matplotlib.pyplot as plt
plt.style.use('ggplot')

loss= model_training_history.history['loss']
val_loss = model_training_history.history['val_loss']

plt.figure()
plt.plot(model_training_history.history['loss'], label="Training Loss")
plt.plot(model_training_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("model leaky relu")
plt.legend()
plt.show()

pred_y = model.predict(x_test_enc)

print(pred_y[0].shape)
print(pred_y[0])
print(np.sum(pred_y[0]))

print(np.argmax(pred_y[0]))


from tensorflow.keras import regularizers

leaky_dropout_l2_model = keras.Sequential([
    layers.InputLayer(input_shape=(10000,)),

    # First dense with L2 + Dropout
    layers.Dense(128, 
                 activation="leaky_relu", 
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),

    # Second dense with L2 + Dropout
    layers.Dense(64, 
                 activation="leaky_relu", 
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),

    # Output layer
    layers.Dense(46, activation="softmax")
])

# Compile
leaky_dropout_l2_model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.05),
    metrics=['accuracy']
)

# Train
leaky_dropout_l2_history = leaky_dropout_l2_model.fit(
    x_train_enc_rest,
    y_train_enc_rest,
    batch_size=512,
    epochs=150,
    validation_data=(x_val, y_val)
)

# Plot
plt.figure()
plt.plot(leaky_dropout_l2_history.history['loss'], label="Training Loss")
plt.plot(leaky_dropout_l2_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Leaky ReLU + L2 Regularisation + Dropout")
plt.legend()
plt.show()
