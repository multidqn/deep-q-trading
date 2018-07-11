import gym
import numpy
import pandas

class SpEnv(gym.Env):
    
    continuous = False

    def __init__(self):
        spTimeserie = pandas.read_csv('sp500.csv')
        dates = spTimeserie.ix[:, 'Date'].tolist()
        timeT = spTimeserie.ix[:, 'Time'].tolist()
        Open = spTimeserie.ix[:, 'Open'].tolist()
        close = spTimeserie.ix[:, 'Close'].tolist()
        values = []
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
            self.history.append({'Date' : dates[i],'Time' : time[i], 'Value' : close[i]-Open[i], 'Open': Open[i] })
        self.minValue = min(values)
        self.maxValue = max(values)
        self.minTime = min(time)
        self.maxTime = max(time)
        #self.low = numpy.array([self.minValue, self.minTime])
        #self.high = numpy.array([self.maxValue, self.maxTime])
        high = numpy.array([numpy.inf]*2)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = gym.spaces.Box(-high, high)
        self.action_space = gym.spaces.Discrete(3) # the action space is just 0,1,2 which means hold,buy,sell
        #self.observation_space = gym.spaces.Box(self.low, self.high)
        # we clean our memory #
        del(dates)            #
        del(Open)             #
        del(close)            #
        del(spTimeserie)      #
        del(values)           #
        #######################

    def step(self, action):
        if self.currentState == 0: # NONE
            self.currentState = action
            reward = 0
            self.done = False
            if action == 1:
                self.currentValue = -self.history[self.currentObservation]['Open']
            elif action == 2:
                self.currentValue = self.history[self.currentObservation]['Open']
            else:
                self.currentValue = 0
        elif self.currentState == 1:# LONG
            if action == 0:
                reward = 0
                self.done = False
            elif action == 2:
                reward = self.currentValue + self.history[self.currentObservation]['Open']
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
                reward = self.currentValue - self.history[self.currentObservation]['Open']
                self.done = True
                self.currentState = 0
            else:
                reward = 0
                self.done = False

        today,tomorrow = self.getTodayTomorrow()
        
        if today != tomorrow:
            self.done = True


        if not self.done:
            self.currentObservation+=1
            self.currentObservation%=self.limit

        currentValue = self.history[self.currentObservation]['Value']
        currentTime = self.history[self.currentObservation]['Time']
        state = [currentValue, currentTime]
        print(type(currentValue))
        print(type(currentTime))
        print(type(state))
        print(len(state))
        assert len(state)==2
        state = numpy.array(state)
        print(state)
        print(state.shape)
        print(len(state))
        print(type(state))
        return state, reward, self.done, {}

    def getCurrentState(self):
        return self.currentState

    def getTodayTomorrow(self):
        return self.history[self.currentObservation]['Date'],self.history[(self.currentObservation+1)%self.limit]['Date']

    def reset(self):
        if self.history[self.currentObservation]['Date'] == self.history[self.currentObservation-1]['Date']:
            currentDate = self.history[self.currentObservation]['Date']
            while currentDate == self.history[self.currentObservation]['Date']:
                self.currentObservation+=1
        self.done = False
        self.currentState = 0
        currentValue = self.history[self.currentObservation]['Value']
        currentTime = self.history[self.currentObservation]['Time']
        state = [currentValue,currentTime]
        print(type(currentValue))
        print(type(currentTime))
        print(type(state))
        print(len(state))
        state = numpy.array(state)
        print(state)
        print(state.shape)
        print(len(state))
        print(type(state))
        return state, 0, self.done, {}

def getEnv():
    return SpEnv()