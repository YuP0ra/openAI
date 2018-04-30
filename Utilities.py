import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """
        :param size: int
        Max number of transitions to store in the buffer.
        When the buffer overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        indices = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        state_t0, actions, rewards, state_t1, done = [], [], [], [], []

        for i in indices:
            obs_t0, action, reward, obs_t1, done = self._storage[i]
            state_t0.append(np.array(obs_t0, copy=False))
            state_t1.append(np.array(obs_t1, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            done.append(done)
        return np.array(state_t0), np.array(actions), np.array(rewards), np.array(state_t1), np.array(dones)
