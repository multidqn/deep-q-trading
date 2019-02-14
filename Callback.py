from rl.callbacks import Callback


class ValidationCallback(Callback):

    def __init__(self):
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0

    def reset(self):
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        

    def on_episode_end(self, episode, logs={}):
        self.episodes+=1
        self.rewardSum+=logs['episode_reward']
        self.coverage+=1 if (logs['episode_reward']!=0) else 0
        self.accuracy+=1 if (logs['episode_reward']>0) else 0

    def getInfo(self):
        acc = 0
        cov = 0
        if self.coverage > 0:
            acc = self.accuracy/self.coverage
        if self.episodes > 0:
            cov = self.coverage/self.episodes
        return self.episodes,cov,acc,self.rewardSum
