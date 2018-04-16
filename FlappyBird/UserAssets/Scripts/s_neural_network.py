from keras.models import Sequential
from keras.layers import Dense

__id__ = 'neural_network'
observation_space, action_space, ep_length, index = 3, 2, 1000, 0


def Start():
    global model, states_buffer, actions_buffer, rewards_buffer, quality_buffer, index

    states_buffer = np.zeros(shape=[ep_length, observation_space], dtype=np.float32)
    actions_buffer = np.zeros(shape=[ep_length, 1], dtype=np.int32)
    rewards_buffer = np.zeros(shape=[ep_length, 1], dtype=np.float32)
    quality_buffer = np.zeros(shape=[ep_length, action_space], dtype=np.float32)

    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(observation_space,)))
    model.add(Dense(action_space))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])


def get_quality(state):
    global model
    return model.predict(state.reshape(-1, observation_space))


def reset_memory():
    global index
    index = 0


def append_memory(s, a, r, q):
    global states_buffer, actions_buffer, rewards_buffer, quality_buffer, index

    if index < ep_length:
        states_buffer[index]  = s
        actions_buffer[index] = a
        rewards_buffer[index] = r
        quality_buffer[index] = q

        index += 1
    else:
        train_patch()
        index = 0


def discount_buffer():
    global actions_buffer, rewards_buffer, quality_buffer, index

    for s in reversed(range(index - 1)):
        rewards_buffer[s] += .98 * rewards_buffer[s + 1]
        quality_buffer[s, actions_buffer[s]] = rewards_buffer[s] + 0.95 * np.max(quality_buffer[s + 1])


def train_patch():
    global states_buffer, quality_buffer, index
    model.fit(x=states_buffer[:index + 1], y=quality_buffer[:index + 1], verbose=0, epochs=20)


def save():
    model.save(filepath='.\\model.h5', overwrite=True)
