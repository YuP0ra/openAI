from keras.models import load_model
import numpy as np
import gym

epsilon = 0.2
ep_length = 500
episodes = 10000

env = gym.make('CartPole-v0')
env._max_episode_steps = ep_length

model = load_model(filepath='.\\model.h5')

current_state, done = env.reset(), False

while not done:
    predictions = model.predict(current_state.reshape(-1, 4))
    chosen_action = np.argmax(predictions, axis=1)[0]
    current_state, r, done, _ = env.step(chosen_action)
    env.render()