#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

#Environment used for spenv 
#gym is the library of videogames used by reinforcement learning
import gym
from gym import spaces
#Numpy is the library to deal with matrices
import numpy
#Pandas is the library used to deal with the CSV dataset
import pandas
#datetime is the library used to manipulate time and date
from datetime import datetime
#Library created by Tonio to merge data used as feature vectors
from mergedDataStructure import MergedDataStructure
#Callback is the library used to show metrics 
import callback

#This is the prefix of the files that will be opened. It is related to the s&p500 stock market datasets
MK = "dax"


class SpEnv(gym.Env):
    #Just for the gym library. In a continuous environment, you can do infinite decisions. 
    #We dont want this because we have just three possible actions.
    continuous = False

    #Observation window is the time window regarding the "hourly" dataset 
    #ensemble variable tells to save or not the decisions at each walk
    def __init__(self, minLimit=None, maxLimit=None, operationCost = 0, observationWindow = 40, ensamble = None, callback = None, isOnlyShort=False, columnName = "iteration-1"):
        #Declare the episode as the first episode
        self.episode=1

        self.isOnlyShort=isOnlyShort
        
        #Open the time series as the hourly dataset of S&P500
        #the input feature vector is composed of data from hours, weeks and days
        #20 from days, 8 from weeks and 40 hours, ending with 68 dimensional feature vectors
        spTimeserie = pandas.read_csv('./datasets/'+MK+'Hour.csv')[minLimit:maxLimit] # opening the dataset
        
        #Converts each column to a list
        Date = spTimeserie.ix[:, 'Date'].tolist()
        Time = spTimeserie.ix[:, 'Time'].tolist()
        Open = spTimeserie.ix[:, 'Open'].tolist()
        High = spTimeserie.ix[:, 'High'].tolist()
        Low = spTimeserie.ix[:, 'Low'].tolist()
        Close = spTimeserie.ix[:, 'Close'].tolist()
        
        #Open the weekly and daily data as a merged data structure
        #Get 20 dimensional vectors (close-open) considering 20 past days and 8 dimensional vectors (close-open) 
        #considering 8 weeks
        self.weekData = MergedDataStructure(delta=8,filename="./datasets/"+MK+"Week.csv")# this DS allows me to obtain previous historical data with different resolution
        self.dayData = MergedDataStructure(delta=20,filename="./datasets/"+MK+"Day.csv")#  with low computational complexity
        
        #Load the data
        self.output=False

        #ensamble is the table of validation and testing
        #If its none, you will not save csvs of validation and testing    
        if(ensamble is not None): # managing the ensamble output (maybe in the wrong way)
            self.output=True
            self.ensamble=ensamble
            self.columnName = columnName
            #self.ensemble is a big table (before file writing) containing observations as lines and epochs as columns
            #each column will contain a decision for each epoch at each date. It is saved later.
            #We read this table later in order to make ensemble decisions at each epoch
            self.ensamble[self.columnName]=0

        #Declare low and high as vectors with -inf values 
        self.low = numpy.array([-numpy.inf])
        self.high = numpy.array([+numpy.inf])

        #Define the space of actions as 3
        #the action space is just 0,1,2 which means hold,long,short
        self.action_space = spaces.Discrete(3) 
        
        #low and high are the minimun and maximum accepted values for this problem
        #Tonio used random values
        #We dont know what are the minimum and maximum values of Close-Open, so we put these values
        self.observation_space = spaces.Box(self.low, self.high, dtype=numpy.float32)

        #The history begins empty
        self.history=[]
        #Set observationWindow = 40
        self.observationWindow = observationWindow
        
        #Set the current observation as 40
        self.currentObservation = observationWindow
        #The operation cost is defined as 
        self.operationCost=operationCost
        #Defines that the environment is not done yet
        self.done = False
        #The limit is the number of open values in the dataset (could be any other value)
        self.limit = len(Open)
        #organizing the dataset as a list of dictionaries 
        for i in range(0,self.limit): 
            self.history.append({'Date' : Date[i],'Time' : Time[i], 'Open': Open[i], 'High': High[i], 'Low': Low[i], 'Close': Close[i]})
        
        #Next observation starts
        self.nextObservation=0
        
        #self.history contains all the hour data. Here we search for the next day 
        while(self.history[self.currentObservation]['Date']==self.history[(self.currentObservation+self.nextObservation)%self.limit]['Date']):
            self.nextObservation+=1
        
        #Initiates the values to be returned by the environment
        self.reward = None
        self.possibleGain = 0
        self.openValue = 0
        self.closeValue = 0
        self.callback=callback


    #This is the action that is done in the environment. 
    #Receives the action and returns the state, the reward and if its done 
    def step(self, action):
        #Initiates the reward, weeklist and daylist
        self.reward=0
        

        ##UNCOMMENT NEXT LINE FOR ONLY SHORT AGENT
        if(self.isOnlyShort):
            action *= 2


        #set the next observation to zero
        self.nextObservation=0
        #Search for the close value for tommorow
        while(self.history[self.currentObservation]['Date']==self.history[(self.currentObservation+self.nextObservation)%self.limit]['Date']):
            #Search for the close error for today
            self.closeValue=self.history[(self.currentObservation+self.nextObservation)%self.limit]['Close']
            self.nextObservation+=1

        #Get the open value for today 
        self.openValue = self.history[self.currentObservation]['Open']

        #Calculate the reward in percentage of growing/decreasing
        self.possibleGain = (self.closeValue - self.openValue)/self.openValue
        #If action is a long, calculate the reward 
        if(action == 1):
            #The reward must be subtracted by the cost of transaction
            self.reward = self.possibleGain-self.operationCost
        #If action is a short, calculate the reward     
        elif(action==2):
            self.reward = (-self.possibleGain)-self.operationCost
        #If action is a hold, no reward     
        else:
            self.reward = 0
        #Finish episode 
        self.done=True


        #Call the callback for the episode
        if(self.callback!=None and self.done):
            self.callback.on_episode_end(action,self.reward,self.possibleGain)
        

        #File of the ensamble (file containing each epoch decisions at each walk) will contain the action for that 
        #day (observation, line) at each epoch (column)
        if(self.output):
            self.ensamble.at[self.history[self.currentObservation]['Date'],self.columnName]=action
        
        
        
        #Return the state, reward and if its done or not
        return self.getObservation(self.history[self.currentObservation]['Date']), self.reward, self.done, {}
        
    #function done when the episode finishes
    #reset will prepare the next state (feature vector) and give it to the agent
    def reset(self):

        if(self.currentObservation<self.observationWindow):
            self.currentObservation=self.observationWindow


        
        self.episode+=1
        
        
        #Shiftting the index for the first hour of the next day
        self.nextObservation=0
        while(self.history[self.currentObservation]['Date']==self.history[(self.currentObservation+self.nextObservation)%self.limit]['Date']):
            self.nextObservation+=1
            #check if the index exceeds the limits
            if((self.currentObservation+self.nextObservation)>=self.limit):
                print("Resetted: episode " + str(self.episode) +"; Index " + str(self.currentObservation+self.nextObservation) + " over the limit (" + str(self.limit) + ")" )
            
        #reset the values used in the step() function
        self.done = False
        self.reward = None
        self.possibleGain = 0
        self.openValue = 0
        self.closeValue = 0

        #Prepapre to get the next observation
        self.currentObservation+=self.nextObservation
        if(self.currentObservation>=self.limit):
            self.currentObservation=self.observationWindow
        
        return self.getObservation(self.history[self.currentObservation]['Date'])


    def getObservation(self, date):

        #Get the dayly information and week information
        #get all the data
        # dayList=self.dayData.get(date)
        # weekList=self.weekData.get(date)

        #Get the previous 40 hours regarding each date
        # currentData = self.history[self.currentObservation-self.observationWindow:self.currentObservation] 

        #The data is finally concatenated here. We concatenate Hours, days and weeks information
        # currentData=self.history[self.currentObservation-self.observationWindow:self.currentObservation]  + self.dayData.get(date) + self.weekData.get(date)

        #Calculates the close minus open 
        #The percentage of growing or decreasing is calculated as CloseMinusOpen
        #This is the input vector
        # closeMinusOpen=list(map(lambda x: (x["Close"]-x["Open"])/x["Open"],self.history[self.currentObservation-self.observationWindow:self.currentObservation]  + self.dayData.get(date) + self.weekData.get(date)))
        
        
        #The state is prepared by the environment, which is simply the feature vector
        return  numpy.array(
            [list(
                map(
                    lambda x: (x["Close"]-x["Open"])/x["Open"],
                        self.history[self.currentObservation-self.observationWindow:self.currentObservation]  + 
                        self.dayData.get(date) + 
                        self.weekData.get(date)))])
    
    def resetEnv(self):
        self.currentObservation=self.observationWindow
        #Resets the episode to 1
        self.episode=1
