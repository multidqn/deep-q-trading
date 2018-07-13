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
        print(spTimeserie.size)
        print(len(dates))
        print(len(timeT))
        print(len(Open))
        print(len(close))
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
        print(len(self.history))
        self.minValue = min(values)
        self.maxValue = max(values)
        self.minTime = min(time)
        self.maxTime = max(time)
        self.low = numpy.array([self.minValue, self.minTime])
        self.high = numpy.array([self.maxValue, self.maxTime])
        self.action_space = gym.spaces.Discrete(3) # the action space is just 0,1,2 which means hold,buy,sell
        self.observation_space = gym.spaces.Box(self.low, self.high)
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


        

        currentValue = self.history[self.currentObservation]['Value']
        currentTime = self.history[self.currentObservation]['Time']
        state = [currentValue, currentTime]
        state = numpy.array(state)
        if not self.done:
            self.currentObservation+=1
            self.currentObservation%=self.limit
        #print(str(self.currentObservation) + "  -  " + str(self.limit))

        return state, reward*50, self.done, {}

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
        state = [currentValue,currentTime]
        state = numpy.array(state)
        #print("RESETTING")
        #print("     -*-" + str(self.currentObservation) + " ---- " + self.history[self.currentObservation]['Date'] + " - " + self.history[self.currentObservation-1]['Date'])
        return state

def getEnv():
    return SpEnv()