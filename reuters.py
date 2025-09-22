
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
plt.show()

pred_y = model.predict(x_test_enc)

print(pred_y[0].shape)
print(pred_y[0])
print(np.sum(pred_y[0]))

print(np.argmax(pred_y[0]))


model2 = keras.Sequential()
model2.add(layers.InputLayer(input_shape= (10000,)))
model2.add(layers.Dense(4, activation='relu'))
model2.add(layers.Dense(4, activation='relu'))
model2.add(layers.Dense(46, activation='softmax'))
model2.summary

model2.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(learning_rate=0.05),
    metrics=['accuracy']
)

model2_training_history = model2.fit(
    x_train_enc_rest,
    y_train_enc_rest,
    batch_size = 512,
    epochs= 120,
    validation_data = (x_val, y_val)
)

plt.figure()
plt.plot(model2_training_history.history['loss'], label="Training Loss")
plt.plot(model2_training_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Model 2 (Shallow)")
plt.legend()
plt.show()


model3 = keras.Sequential()
model3.add(layers.InputLayer(input_shape=(10000,))) 

model3.add(layers.Dense(64, activation="relu"))
model3.add(layers.Dropout(0.2))

model3.add(layers.Dense(64, activation="relu"))
model3.add(layers.Dropout(0.2))

model3.add(layers.Dense(46, activation="softmax")) 
model3.summary()

model3.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.05), 
    metrics=['accuracy']
)


model3_training_history = model3.fit(
    x_train_enc_rest,
    y_train_enc_rest,
    batch_size = 512,
    epochs = 120, 
    validation_data = (x_val, y_val)
)


plt.figure()
plt.plot(model3_training_history.history['loss'], label="Training Loss")
plt.plot(model3_training_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Model 3 (Dropout 0.2)")
plt.legend()
plt.show()

model4 = keras.Sequential()
model4.add(layers.InputLayer(input_shape=(10000,)))

model4.add(layers.Dense(64, activation="relu"))
model4.add(layers.Dropout(0.5))

model4.add(layers.Dense(64, activation="relu"))
model4.add(layers.Dropout(0.5))

model4.add(layers.Dense(46, activation="softmax"))
model4.summary()

model4.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.05),
    metrics=['accuracy']
)

model4_training_history = model4.fit(
    x_train_enc_rest,
    y_train_enc_rest, 
    batch_size = 512,
    epochs = 40,
    validation_data = (x_val, y_val)
)
plt.figure()
plt.plot(model4_training_history.history['loss'], label="Training Loss")
plt.plot(model4_training_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Model 4 (Dropout 0.5)")
plt.legend()
plt.show()


plt.plot(model3_training_history.history['loss'], '--', label="D0.2 -Train loss")
plt.plot(model3_training_history.history['val_loss'], label="D0.2 - Val Loss")

plt.plot(model4_training_history.history['loss'], '--', label="D0.2 -Train loss")
plt.plot(model4_training_history.history['val_loss'], label="D0.2 - Val Loss")

plt.plot(model_training_history.history['loss'], '--', label="D0.2 -Train loss")
plt.plot(model_training_history.history['val_loss'], label="D0.2 - Val Loss")

plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

plt.plot(model3_training_history.history['accuracy'], '--', label="D0.2 - Train Acc")
plt.plot(model4_training_history.history['accuracy'], '--', label="D0.5 - Train Acc")
plt.plot(model_training_history.history['accuracy'], '--', label="Original Train Acc")

plt.ylabel('Acc')
plt.xlabel('epochs')
plt.legend()
plt.show()


bn_model = keras.Sequential()
bn_model.add(layers.InputLayer(input_shape=(10000,)))

bn_model.add(layers.Dense(64))
bn_model.add(layers.BatchNormalization())
bn_model.add(layers.ReLU())

bn_model.add(layers.Dense(64))
bn_model.add(layers.BatchNormalization())
bn_model.add(layers.ReLU())

bn_model.add(layers.Dense(46, activation='softmax'))
bn_model.summary()

bn_model.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(learning_rate=0.05),
    metrics=['accuracy']
)

bn_model_training_history = bn_model.fit(
    x_train_enc_rest,
    y_train_enc_rest,
    batch_size = 512,
    epochs = 80, 
    validation_data = (x_val, y_val)
)


plt.figure()
plt.plot(bn_model_training_history.history['loss'], label="Training Loss")
plt.plot(bn_model_training_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("BN Model")
plt.legend()
plt.show()


bn_l2_model = keras.Sequential()
bn_l2_model.add(layers.InputLayer(input_shape=(10000,)))

bn_l2_model.add(layers.Dense(64, kernel_regularizer='l2'))
bn_l2_model.add(layers.BatchNormalization())
bn_l2_model.add(layers.ReLU())

bn_l2_model.add(layers.Dense(64, kernel_regularizer='l2'))
bn_l2_model.add(layers.BatchNormalization())
bn_l2_model.add(layers.ReLU())

bn_l2_model.add(layers.Dense(46, activation='softmax'))
bn_l2_model.summary()

bn_l2_model.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(learning_rate=0.05),
    metrics=['accuracy']
)

bn_l2_model_training_history = bn_l2_model.fit(
    x_train_enc_rest,
    y_train_enc_rest,
    batch_size = 512,
    epochs = 100,
    validation_data= (x_val, y_val)
)

plt.figure()
plt.plot(bn_model_training_history.history['loss'], label="Training Loss")
plt.plot(bn_model_training_history.history['val_loss'], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("BN Model2")
plt.legend()
plt.show()


from tensorflow.keras import layers, activations

# Baseline with ReLU
baseline_model = keras.Sequential([
    layers.InputLayer(input_shape=(10000,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

# Variant with LeakyReLU
leaky_model = keras.Sequential([
    layers.InputLayer(input_shape=(10000,)),
    layers.Dense(64),
    layers.LeakyReLU(alpha=0.1),   # add separately
    layers.Dense(64),
    layers.LeakyReLU(alpha=0.1),
    layers.Dense(46, activation="softmax")
])

# Variant with ELU
elu_model = keras.Sequential([
    layers.InputLayer(input_shape=(10000,)),
    layers.Dense(64, activation="elu"),
    layers.Dense(64, activation="elu"),
    layers.Dense(46, activation="softmax")
])
