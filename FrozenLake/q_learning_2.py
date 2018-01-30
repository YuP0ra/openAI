import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()


inputs1 = tf.placeholder(shape=[None,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)


nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
loss = tf.losses.mean_squared_error(labels=nextQ, predictions=Qout)
trainer = tf.train.AdamOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 20000

jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    rAll = 0

    for i in range(num_episodes):
        targetQ = np.zeros((99, 4))
        inp = np.zeros((99, 16))
        s = env.reset()
        d = False
        j = 0

        while j < 98:
            j += 1

            a,allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            s1, r, d, _ = env.step(a[0])

            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})

            maxQ1 = np.max(Q1)
            targetQ[j] = allQ
            targetQ[j, a[0]] = r + y * maxQ1
            inp[j] = np.identity(16)[s]

            rAll += r
            s = s1
            if d is True:
                e = 1./((i/50) + 10)
                sess.run([updateModel, W], feed_dict={inputs1: inp[:j+1], nextQ: targetQ[:j+1]})
                break

        if i % 100 == 0:
            print("Percent of successful episodes: ", rAll)
            rAll = 0
