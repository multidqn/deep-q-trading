from rl.callbacks import Callback


class ValidationCallback(Callback):

    def __init__(self):
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortPrecision =0
        self.longPrecision =0

    def reset(self):
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        

    def on_episode_end(self, action, reward):
        self.episodes+=1
        self.rewardSum+=reward
        self.coverage+=1 if (action != 0) else 0
        self.accuracy+=1 if (reward >= 0 and action != 0) else 0
        self.short +=1 if(action == 2) else 0
        self.long +=1 if(action == 1) else 0
        self.shortPrecision +=1 if(action == 2 and reward >=0) else 0
        self.longPrecision +=1 if(action == 1 and reward >=0) else 0
        
        

    def getInfo(self):
        acc = 0
        cov = 0
        if self.coverage > 0:
            acc = self.accuracy/self.coverage
        if self.episodes > 0:
            cov = self.coverage/self.episodes
            short = self.short/self.episodes
            long = self.long/self.episodes
        if self.short > 0:
            shortP = self.shortPrecision/self.short
        if self.long > 0:
            longP = self.longPrecision/self.long
        return self.episodes,cov,acc,self.rewardSum,long,short,longP,shortP
