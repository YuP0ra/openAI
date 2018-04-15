import tensorflow as tf
import numpy as np
import gym

learning_rate = 0.001
epsilon = 0.2
episodes = 100000
ep_length = 500

tf.reset_default_graph()

states_holder = tf.placeholder(tf.float32, shape=[None, 4])
q_holder = tf.placeholder(tf.float32, shape=[None, 2])

weights_0 = tf.Variable(tf.random_uniform([4, 64], 0, 0.01))
weights_1 = tf.Variable(tf.random_uniform([64, 2], 0, 0.01))
biases_0 = tf.Variable(tf.random_uniform([64], 0, 0.01))
biases_1 = tf.Variable(tf.random_uniform([2], 0, 0.01))

out_0 = tf.nn.relu(tf.matmul(states_holder, weights_0) + biases_0)
out_1 = tf.matmul(out_0, weights_1) + biases_1

loss = tf.losses.mean_squared_error(predictions=out_1, labels=q_holder)
update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

env = gym.make('CartPole Tensorflow-v0')
env._max_episode_steps = ep_length

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    states_buffer = np.zeros(shape=[ep_length, 4])
    actions_buffer = np.zeros(shape=[ep_length, 1], dtype=np.int32)
    rewards_buffer = np.zeros(shape=[ep_length, 1])
    q_buffer = np.zeros(shape=[ep_length, 2])

    fi_x, fi_y = np.zeros(shape=[1, 4]), np.zeros(shape=[1, 2])

    r_all = 0
    for e in range(episodes):
        s = env.reset()

        for j in range(ep_length):
            a_p = sess.run([out_1], feed_dict={states_holder: [s]})[0][0]

            if np.random.ranf() < 0.2:
                chosen_action = env.action_space.sample()
            else:
                chosen_action = np.argmax(a_p)

            states_buffer[j] = s
            q_buffer[j] = a_p
            actions_buffer[j] = chosen_action

            next_state, rewards_buffer[j], done, _ = env.step(chosen_action)

            r_all += rewards_buffer[j]
            rewards_buffer[j] = rewards_buffer[j] / (1. + np.abs(s[0]))
            if done or j == ep_length - 1:
                s = j
                break
            else:
                s = next_state

        q_buffer[s] = np.zeros(2)
        for t in reversed(range(s)):
            rewards_buffer[t] = rewards_buffer[t] + 0.95 * rewards_buffer[t + 1]
            q_buffer[t, actions_buffer[t]] = (1 - epsilon) * q_buffer[t, actions_buffer[t]] + epsilon * (rewards_buffer[t] + 0.95 * np.max(q_buffer[t + 1]))

            sess.run([update], feed_dict={states_holder: states_buffer[:s+1], q_holder: q_buffer[:s+1]})

        if e % 99 == 0:
            print(r_all)
            r_all = 0
