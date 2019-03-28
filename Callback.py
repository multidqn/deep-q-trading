from rl.callbacks import Callback


class ValidationCallback(Callback):

    def __init__(self):
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0

    def reset(self):
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0
        

    def on_episode_end(self, action, reward, market):
        self.episodes+=1
        self.rewardSum+=reward
        self.coverage+=1 if (action != 0) else 0
        self.accuracy+=1 if (reward >= 0 and action != 0) else 0
        self.short +=1 if(action == 2) else 0
        self.long +=1 if(action == 1) else 0
        self.shortAcc +=1 if(action == 2 and reward >=0) else 0
        self.longAcc +=1 if(action == 1 and reward >=0) else 0
        if(market>0):
            self.marketRise+=1
            self.longPrec+=1 if(action == 1) else 0
        elif(market<0):
            self.marketFall+=1
            self.longPrec+=1 if(action == 2) else 0

    def getInfo(self):
        acc = 0
        cov = 0
        short = 0
        long = 0
        longAcc = 0
        shortAcc = 0
        longPrec = 0
        shortPrec = 0
        if self.coverage > 0:
            acc = self.accuracy/self.coverage
        if self.episodes > 0:
            cov = self.coverage/self.episodes
            short = self.short/self.episodes
            long = self.long/self.episodes
        if self.short > 0:
            shortAcc = self.shortAcc/self.short
        if self.long > 0:
            longAcc = self.longAcc/self.long
        if self.marketRise > 0:
            longPrec = self.longPrec/self.marketRise
        if self.marketFall > 0:
            shortPrec = self.shortPrec/self.marketFall
            
        return self.episodes,cov,acc,self.rewardSum,long,short,longAcc,shortAcc,longPrec,shortPrec
