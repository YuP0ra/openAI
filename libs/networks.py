import tensorflow as tf


class QNetwork:
    def __init__(self,
                 architecture,
                 stddev=1e-3,
                 learning_rate=1e-3
                 ):

        self.sess = None
        self.__architecture = architecture

        self.x_holder = tf.placeholder(tf.float32, [None, architecture[0]])
        self.y_holder = tf.placeholder(tf.float32, [None, architecture[-1]])

        self.__w0 = tf.Variable(tf.random_normal([architecture[0], architecture[1]], stddev=stddev))
        self.__w1 = tf.Variable(tf.random_normal([architecture[1], architecture[2]], stddev=stddev))
        self.__w2 = tf.Variable(tf.random_normal([architecture[2], architecture[3]], stddev=stddev))

        self.__b0 = tf.Variable(tf.random_normal([architecture[1]], stddev=stddev))
        self.__b1 = tf.Variable(tf.random_normal([architecture[2]], stddev=stddev))
        self.__b2 = tf.Variable(tf.random_normal([architecture[3]], stddev=stddev))

        self.__optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.__loss = tf.losses.mean_squared_error(predictions=self.__feed_forward(), labels=self.y_holder)
        self.__update = self.__optimizer.minimize(self.__loss)

    def attach_session(self, sess):
        self.sess = sess

    def __feed_forward(self,):
        out_1 = tf.matmul(self.x_holder, self.__w0) + self.__b0
        out_2 = tf.matmul(out_1, self.__w1) + self.__b1
        out_3 = tf.matmul(out_2, self.__w2) + self.__b2
        return out_3

    def predict(self, input_data):
        output = self.sess.run([self.__feed_forward()], feed_dict={self.x_holder: input_data})
        return output

    def back_prop(self, data, labels, iterations=1):
        for i in range(iterations):
            self.sess.run([self.__update], feed_dict={self.x_holder: data, self.y_holder: labels})
