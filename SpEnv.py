import gym
import numpy
import pandas

class SpEnv(gym.Env):
    
    continuous = False

    def __init__(self, minLimit=None, maxLimit=None, verbose=False, operationCost = 0):
        self.verbose = verbose
        self.operationCost = operationCost
        if(verbose):
            print("Episode,Date,Reward,Operation,PriceIn,TimeIn,PriceOut,TimeOut,Steps,Capital,AvgRw,MaxRw,MinRw,hit,miss,hold,long,short")
        spTimeserie = pandas.read_csv('sp500.csv')[minLimit:maxLimit]
        dates = spTimeserie.ix[:, 'Date'].tolist()
        timeT = spTimeserie.ix[:, 'Time'].tolist()
        Open = spTimeserie.ix[:, 'Open'].tolist()
        close = spTimeserie.ix[:, 'Close'].tolist()
        #print(spTimeserie.size)
        self.operation = 0
        self.nbHold = 0
        self.nbLong = 0
        self.nbShort = 0
        self.nbHit = 0
        self.nbMiss = 0
        self.totalReward = 0
        self.timeFirst = ""
        self.timeSecond = ""
        self.priceFirst = 0
        self.priceSecond = 0
        self.episodeReward = 0
        self.episodeSteps = 0
        self.maxEp = 0
        self.minEp = 0
        values = []
        self.episode = 1
        self.done = False
        self.currentState = 0 # (0 = nothing) (1 = long) (2 = short)
        self.currentValue = 0
        self.currentObservation = 0 # number of executed steps
        self.limit = len(Open) # the end of my timeserie
        self.history = [] # the timeserie itself
        time = []
        for t in timeT: # here i parse all the daytimes into seconds
            time.append(sum(x * int(t) for x, t in zip([3600, 60], t.split(":")))) 
        for i in range(0,self.limit): # i use this to create my states (as dictionaries)
            values.append(close[i]-Open[i])
            self.history.append({'Date' : dates[i],'TimeT' : timeT[i],'Time' : time[i], 'Value' : close[i]-Open[i], 'Open': Open[i] })
        #print(len(self.history))
        self.minValue = min(values)
        self.maxValue = max(values)
        self.minTime = min(time)
        self.maxTime = max(time)
        self.low = numpy.array([self.minValue, self.minTime, 0])
        self.high = numpy.array([self.maxValue, self.maxTime, 2])
        self.action_space = gym.spaces.Discrete(3) # the action space is just 0,1,2 which means hold,buy,sell
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=numpy.float32)
        # we clean our memory #
        del(dates)            #
        del(Open)             #
        del(close)            #
        del(spTimeserie)      #
        del(time)             #
        del(timeT)            #
        del(values)           #
        #######################

    def step(self, action):
        if self.currentState == 0: # NONE
            self.operation = action
            self.currentState = action
            reward = 0
            self.done = False
            if action == 1:
                self.currentValue = -self.history[self.currentObservation]['Open']
                self.priceFirst = self.history[self.currentObservation]['Open']
                self.timeFirst = self.history[self.currentObservation]['TimeT']
                reward-= self.operationCost
            elif action == 2:
                self.currentValue = self.history[self.currentObservation]['Open']
                self.priceFirst = self.history[self.currentObservation]['Open']
                self.timeFirst = self.history[self.currentObservation]['TimeT']
                reward-= self.operationCost
            else:
                self.currentValue = 0
                
        elif self.currentState == 1:# LONG
            if action == 0:
                reward = 0
                self.done = False
            elif action == 2:
                reward = (self.currentValue + self.history[self.currentObservation]['Open'])*50 - self.operationCost
                self.priceSecond = self.history[self.currentObservation]['Open']
                self.timeSecond = self.history[self.currentObservation]['TimeT']
                self.done = True
                self.currentState = 0
            else:
                reward = 0
                self.done = False
            
        else: # SHORT
            if action == 0:
                reward = 0
                self.done = False
            elif action == 1:
                reward = (self.currentValue - self.history[self.currentObservation]['Open'])*50 - self.operationCost
                self.priceSecond = self.history[self.currentObservation]['Open']
                self.timeSecond = self.history[self.currentObservation]['TimeT']
                self.done = True
                self.currentState = 0
            else:
                reward = 0
                self.done = False

        today,tomorrow = self.getTodayTomorrow()
        
        if today != tomorrow:
            self.done = True


        

        currentValue = self.history[self.currentObservation]['Value']
        currentTime = self.history[self.currentObservation]['Time']
        state = [currentValue, currentTime, self.currentState]
        state = numpy.array(state)
        if not self.done:
            self.currentObservation+=1
            self.currentObservation%=self.limit
        #print(str(self.currentObservation) + "  -  " + str(self.limit))
        self.episodeReward += reward
        self.totalReward += reward
        self.episodeSteps += 1
        if self.minEp > reward:
            self.minEp = reward
        if self.maxEp < reward:
            self.maxEp = reward
        return state, reward, self.done, {}

    def getCurrentState(self):
        return self.currentState

    def getTodayTomorrow(self):
        Next = self.currentObservation + 1
        Next %= self.limit
        return self.history[self.currentObservation]['Date'],self.history[Next]['Date']

    def reset(self):
        if self.history[self.currentObservation]['Date'] == self.history[self.currentObservation-1]['Date']:
            currentDate = self.history[self.currentObservation]['Date']
            while currentDate == self.history[self.currentObservation]['Date']:
                self.currentObservation+=1
                self.currentObservation%=self.limit
        self.done = False
        self.currentState = 0
        currentValue = self.history[self.currentObservation]['Value']
        currentTime = self.history[self.currentObservation]['Time']
        state = [currentValue,currentTime,self.currentState]
        state = numpy.array(state)

        if self.episodeReward > 0:
            self.nbHit += 1
        elif self.episodeReward < 0:
            self.nbMiss += 1
        
        if self.operation == 0:
            self.nbHold += 1
        elif self.operation == 1:
            self.nbLong += 1
        else:
            self.nbShort += 1

        if(self.verbose):
            Episode = self.episode
            Date = self.history[self.currentObservation]['Date']
            Reward = self.episodeReward
            Operation = self.operation
            PriceIn = self.priceFirst
            TimeIn = self.timeFirst
            PriceOut = self.priceSecond
            TimeOut = self.timeSecond
            Steps = self.episodeSteps
            Capital = self.totalReward
            AvgRw = self.totalReward/self.episode
            MaxRw = self.maxEp
            MinRw = self.minEp
            print(str(Episode)+","+str(Date)+","+str(Reward)+","+str(Operation)+","+str(PriceIn)+","+str(TimeIn)+","+str(PriceOut)+","+str(TimeOut)+","+str(Steps)+","+str(Capital)+","+str(AvgRw)+","+str(MaxRw)+","+str(MinRw)+","+str(int(100*(self.nbMiss/self.episode)))+","+str(int(100*(self.nbHit/self.episode)))+","+str(int(100*(self.nbHold/self.episode)))+","+str(int(100*(self.nbLong/self.episode)))+","+str(int(100*(self.nbShort/self.episode))))
        self.episodeReward = 0
        self.episodeSteps = 0
        self.timeFirst = ""
        self.timeSecond = ""
        self.operation = 0
        self.episode += 1
        self.maxEp = 0
        self.minEp = 0
        self.priceFirst = 0
        self.priceSecond = 0
        return state

    def getProfit(self):
        if(self.currentState == 1):
            return (self.history[self.currentObservation]['Open']-self.priceFirst)*50
        elif(self.currentState == 2):
            return (self.priceFirst-self.history[self.currentObservation]['Open'])*50
        else:
            return 0

def getEnv(minLimit=None, maxLimit=None, verbose=False):
    return SpEnv(minLimit,maxLimit,verbose)
