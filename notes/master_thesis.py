# Imports
import pandas as pd
from keras.models import Sequential
from keras import Input
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
from keras.layers import SimpleRNN
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Reshape, Conv2D, Flatten

# Data preparation

dataset_path = ''
data = pd.read_csv(dataset_path)
data = data.sample(n=1000000, random_state=42)
data['puzzle'] = data['puzzle'].str.replace('.', '0').apply(lambda x: [int(char) for char in x])
data['solution'] = data['solution'].apply(lambda x: [int(char) for char in x])
data.head(10)

df = data[['puzzle', 'solution']]
df.head(10)

puzzle = np.array(df['puzzle'].tolist())
solution = np.array(df['solution'].tolist())

puzzle = puzzle.reshape(-1, 9, 9)
solution = solution.reshape(-1, 9, 9)

# Training models

# LSTM

print('######')
print('lstm1')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('lstm1.keras')

#################################################################################################################
print('######')
print('lstm2')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('lstm2.keras')

################################################################################################################
print('######')
print('lstm3')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('lstm3.keras')

#################################################################################################################
print('######')
print('lstm4')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('lstm4.keras')

# SimpleRNN

print('######')
print('srnn1')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    SimpleRNN(81, input_shape=(9, 9)),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('srnn1.keras')

# #################################################################################################################
print('######')
print('srnn2')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    SimpleRNN(81, input_shape=(9, 9)),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('srnn2.keras')

################################################################################################################
print('######')
print('srnn3')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    SimpleRNN(81, input_shape=(9, 9)),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('srnn3.keras')

#################################################################################################################
print('######')
print('srnn4')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    SimpleRNN(81, input_shape=(9, 9)),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('srnn4.keras')

# CNN

print('######')
print('cnn1')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    Input(shape=(9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('cnn1.keras')

#################################################################################################################
print('######')
print('cnn2')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    Input(shape=(9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('cnn2.keras')

################################################################################################################
print('######')
print('cnn3')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    Input(shape=(9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('cnn3.keras')

#################################################################################################################

print('######')
print('cnn4')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    Input(shape=(9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('cnn4.keras')

# Mixed models

# LSTM first 64 epochs

print('######')
print('LC64')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9),
    Reshape((9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('lc64.keras')

# CNN first 64 epochs

print('######')
print('CL64')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    Input(shape=(9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9)),
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9),
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('cl64.keras')

# LSTM first 128 epochs

print('######')
print('LC128')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9),
    Reshape((9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9))
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=128, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('lc128.keras')

# CNN first 128 epochs

print('######')
print('CL128')
print('######')

tensorboard_callback = TensorBoard('./logs', histogram_freq=1)

model = Sequential([
    Input(shape=(9, 9, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(81),
    Reshape((9, 9)),
    LSTM(64, input_shape=(9, 9), return_sequences=True),
    Dense(9),
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

model.fit(puzzle, solution, epochs=10, batch_size=128, validation_split=0.2, callbacks=[tensorboard_callback])

model.save('cl128.keras')
