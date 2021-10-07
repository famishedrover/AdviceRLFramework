class ReplayBuffer(object):
    def __init__(self, capacity, batch_size=32, gamma=0.1):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.batch_size = batch_size


    def push(self, transition):
        state, action, reward, next_state, done, info = transition

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def extend(self, transitions):
        for i, transition in enumerate(transitions):
            self.push(transition)

    def sample(self, batch_size=None):
        batch_size=self.batch_size if batch_size is not None else batch_size

        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
