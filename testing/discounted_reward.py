from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np

# define the data
data = np.arange(100).reshape(-1, 1) * 10


def custom_loss(network_input_as_y, network_output_as_prediction):
    # define here how you want your loss to do
    target_value = tf.sqrt(network_input_as_y)

    # mean square error
    diff_2 = tf.square(network_output_as_prediction - target_value)

    # final loss, must be positive as the network will try to minimize it
    return np.sum(diff_2)


# define the model and it's design.
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=1))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(1))

model.compile(loss=custom_loss, optimizer='adam')

model.fit(x=data, y=data, epochs=3000, verbose=0)

test_data = np.array([154, 1254]).reshape(-1, 1)
final = model.predict(test_data)
print(final)
