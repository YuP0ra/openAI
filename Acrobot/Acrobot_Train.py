from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gym

"""------------------------------ Variables ------------------------------"""
epsilon = 0.9
ep_length = 100
episodes = 1500

env = gym.make('Acrobot-v1')
env._max_episode_steps = ep_length
observation_space, action_space = env.observation_space.shape[0], env.action_space.n


"""------------------------------ Model ------------------------------"""
# For a single-input model with 2 classes (binary classification):
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(observation_space,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(action_space))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])


"""------------------------------ Train ------------------------------"""
for i in range(episodes):
    states_buffer = np.zeros(shape=[ep_length, observation_space], dtype=np.float32)
    actions_buffer = np.zeros(shape=[ep_length, 1], dtype=np.int32)
    rewards_buffer = np.zeros(shape=[ep_length, 1], dtype=np.float32)
    quality_buffer = np.zeros(shape=[ep_length, action_space], dtype=np.float32)

    current_state = env.reset()
    for j in range(ep_length):
        predictions = model.predict(current_state.reshape(-1, observation_space))

        if np.random.ranf() < epsilon:
            chosen_action = env.action_space.sample()
        else:
            chosen_action = np.argmax(predictions, axis=1)[0]

        states_buffer[j] = current_state
        actions_buffer[j] = chosen_action
        quality_buffer[j] = predictions

        current_state, rewards_buffer[j], done, _ = env.step(chosen_action)

        if done:
            current_state = j
            break

    for k in reversed(range(current_state - 1)):
        rewards_buffer[k] += .95 * rewards_buffer[k + 1]
        quality_buffer[k, actions_buffer[k]] = rewards_buffer[k] + 0.95 * np.max(quality_buffer[k + 1])

    model.fit(x=states_buffer[:current_state + 1], y=quality_buffer[:current_state + 1], verbose=0, epochs=20)

    if i % 100 == 0:
        epsilon *= 0.9
        print('Completed: \t{0} out of: \t{1}'.format(i, episodes))

model.save(filepath='.\\model.h5', overwrite=True)

