import gym
import numpy
import pandas

class SpEnv(gym.Env):
    
    def __init__(self):
        spTimeserie = pandas.read_csv('sp500.csv')
        dates = spTimeserie.ix[:, 'Date'].tolist()
        timeT = spTimeserie.ix[:, 'Time'].tolist()
        open = spTimeserie.ix[:, 'Open'].tolist()
        close = spTimeserie.ix[:, 'Close'].tolist()
        values = []
        self.currentState = 0 # (0 = nothing) (1 = long) (2 = short)
        self.currentObservation = 0 # number of executed steps
        self.limit = len(open) # the end of my timeserie
        self.history = [] # the timeserie itself
        time = []
        for t in timeT: # here i parse all the daytimes into seconds
            time.append(sum(x * int(t) for x, t in zip([3600, 60], t.split(":")))) 
        for i in range(0,self.limit): # i use this to create my states (as dictionaries)
            values.append(close[i]-open[i])
            self.history.append({'Date' : dates[i],'Time' : time[i], 'Value' : close[i]-open[i] })
        self.minValue = min(values)
        self.maxValue = max(values)
        self.minTime = min(time)
        self.maxTime = max(time)
        self.low = numpy.array([self.minValue, self.minTime])
        self.high = numpy.array([self.maxValue, self.maxTime])
        self.action_space = gym.spaces.Discrete(3) # the action space is just 0,1,2 which means hold,buy,sell
        self.observation_space = gym.spaces.Tuple((# in the observaction space we need a mix of the internal state which can be: none, long, short and the current market status
            gym.spaces.Box(self.low, self.high, dtype=numpy.float32),
            gym.spaces.Discrete(3)))
        # we clean our memory #
        del(dates)            #
        del(open)             #
        del(close)            #
        del(spTimeserie)      #
        del(values)           #
        #######################

    def step(self, action):
        # capire cosa fa l'agente
        # se ho fatto HOLD, non faccio niente
        # se ho fatto BUY, mi segno il nuovo INTERNAL_STATE e mi segno quanto ho speso
        # se ho fatto SELL, mi segno il nuovo INTERNAL_STATE e mi segno quanto ho speso
        # se avevo già fatto un'azione (LONG, SHORT), mi calcolo il reward
        # lo stato sarà strutturato:
        currentValue = self.history[self.currentObservation]['Value']
        currentTime = self.history[self.currentObservation]['Time']
        reward = 0
        done = False
        return (numpy.array([currentValue,currentTime]) , numpy.array([self.getCurrentState()]), reward, done, {})

    def getCurrentState(self):
        return self.currentState

    def reset(self):
        return