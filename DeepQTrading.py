#Imports the SPEnv library, which will perform the Agent actions itself
from SpEnv import SpEnv

#Callback used to print results at each episode
from Callback import ValidationCallback

#Keras library for the NN considered
from keras.models import Sequential

#Keras libraries for layers, activation and optimizer used
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam

#RL Agent 
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

#Mathematical operation used later
from math import floor

#Library to manipulate the dataset in a csv file
import pandas as pd

#Library used to manipulate time
import datetime

#Library used 
import telegram

MK="sp500"

class DeepQTrading:
    
    #Class constructor
    #model: Keras model considered
    #Explorations is a vector containing the policy of the probability of random predictions plus how many epochs will be 
    # runned by the algorithm (we run the algorithm several times-several iterations)  
    #trainSize: size of the training set
    #validationSize: size of the validation set
    #testSize: size of the testing set 
    #outputFile: name of the file to print results
    #begin: Initial date
    #end: final date
    #nbActions: number of decisions (0-Hold 1-Long 2-Short) 
    #nOutput is the number of walks. Tonio put 20 but it is 5 walks in reality.  
    #operationCost: Price for the transaction
    #telegramToken: token used for the bot that will send messages
    #telegramChatID: ID of messager receiver in Telegram
    #ensemble.py runs the ensemble
    def __init__(self, model, explorations, trainSize, validationSize, testSize, outputFile, begin, end, nbActions, nOutput=1, operationCost=0,telegramToken="",telegramChatID=""):
        
        #If the telegram token for the bot and the telegram id of the receiver are empty, try to send a message 
        #otherwise print error
        if(telegramToken!="" and telegramChatID!=""):
            self.chatID=telegramChatID
            self.telegramOutput=True
            try:
                self.bot = telegram.Bot(token=telegramToken)
            except:
                print("Error with Telegram Bot")
        
        #If they are not empty, prepare the bot to send messages
        else:
            self.telegramOutput=True

        #Define the policy, explorations, actions and model as received by parameters
        self.policy = EpsGreedyQPolicy()
        self.explorations=explorations
        self.nbActions=nbActions
        self.model=model

        #Define the memory
        self.memory = SequentialMemory(limit=10000, window_length=1)

        #Instantiate the agent with parameters received
        self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                    enable_double_dqn=True,enable_dueling_network=True)
        
        #Compile the agent with the adam optimizer and with the mean absolute error metric
        self.agent.compile(Adam(lr=1e-3), metrics=['mae'])

        #Save the weights of the agents in the q.weights file
        #Save random weights
        self.agent.save_weights("q.weights", overwrite=True)

        #Define the current starting point as the initial date
        self.currentStartingPoint = begin

        #Define the training, validation and testing size as informed by the call
        #Train: five years
        #Validation: 6 months
        #Test: 6 months
        self.trainSize=trainSize
        self.validationSize=validationSize
        self.testSize=testSize
        
        #The walk size is simply summing up the train, validation and test sizes
        self.walkSize=trainSize+validationSize+testSize
        
        #Define the ending point as the final date (January 1st of 2010)
        self.endingPoint=end

        #Read the hourly dataset
        #We join data from different files
        #Here read hour 
        self.dates= pd.read_csv('./dataset/'+MK+'Hour.csv')

        #Read the hourly dataset
        self.sp = pd.read_csv('./dataset/'+MK+'Hour.csv')
        #Convert the pandas format to date and time format
        self.sp['Datetime'] = pd.to_datetime(self.sp['Date'] + ' ' + self.sp['Time'])
        #Set an index to Datetime on the pandas loaded dataset. Register will be indexes through this value
        self.sp = self.sp.set_index('Datetime')
        #Drop Time and Date from the Dataset
        self.sp = self.sp.drop(['Time','Date'], axis=1)
        #Just the index will be important, because date and time will be used to define the train, validation and test 
        #for each walk
        self.sp = self.sp.index

        #Receives the operation cost which is 0
        #Operation cost is the cost for long and short. It is defined as zero
        self.operationCost = operationCost
        
        #Call the callback for training, validation and test in order to show the results for each episode 
        self.trainer=ValidationCallback()
        self.validator=ValidationCallback()
        self.tester=ValidationCallback()
        
        #Initiate the output file
        self.outputFile=[]
        
        #Write in the file
        for i in range(0,nOutput):
            
          
            self.outputFile.append(open(outputFile+str(i+1)+".csv", "w+"))

            #Write the fields in the file
            self.outputFile[i].write(
            "Iteration,"+
            "trainAccuracy,"+
            "trainCoverage,"+
            "trainReward,"+
            "trainLong%,"+
            "trainShort%,"+
            "trainLongAcc,"+
            "trainShortAcc,"+
            "trainLongPrec,"+
            "trainShortPrec,"+

            "validationAccuracy,"+
            "validationCoverage,"+
            "validationReward,"+
            "validationLong%,"+
            "validationShort%,"+
            "validationLongAcc,"+
            "validationShortAcc,"+
            "validLongPrec,"+
            "validShortPrec,"+
            
            "testAccuracy,"+
            "testCoverage,"+
            "testReward,"+
            "testLong%,"+
            "testShort%,"+
            "testLongAcc,"+
            "testShortAcc,"+
            "testLongPrec,"+
            "testShortPrec\n")
        

    def run(self):

        #Initiate the training, 
        trainEnv=validEnv=testEnv=" "

        iteration=-1

        #While we did not pass through all the dates (i.e., while all the walks were not finished)
        #walk size is train+validation+test size
        #currentStarting point begins with begin date
        while(self.currentStartingPoint+self.walkSize <= self.endingPoint):

            #Iteration is a walks
            iteration+=1

            #Send to the receiver the current walk
            if(self.telegramOutput):
                self.bot.send_message(chat_id=self.chatID, text="Walk "+str(iteration + 1 )+" started.")
            
            #Empty the memory and agent
            del(self.memory)
            del(self.agent)

            #Define the memory and agent
            #Memory is Sequential
            self.memory = SequentialMemory(limit=10000, window_length=1)
            #Agent is initiated as passed through parameters
            self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                    enable_double_dqn=True,enable_dueling_network=True)
            #Compile the agent with Adam initialization
            self.agent.compile(Adam(lr=1e-3), metrics=['mae'])
            
            #Load the weights saved before in a random way if it is the first time
            self.agent.load_weights("q.weights")
            
            ########################################TRAINING STAGE########################################################
            
            #The TrainMinLimit will be loaded as the initial date at the beginning, and will be updated later.
            #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date    
            trainMinLimit=None
            while(trainMinLimit is None):
                try:
                    trainMinLimit = self.sp.get_loc(self.currentStartingPoint)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

            #The TrainMaxLimit will be loaded as the interval between the initial date plus the training size.
            #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date    
            trainMaxLimit=None
            while(trainMaxLimit is None):
                try:
                    trainMaxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)
            
            ########################################VALIDATION STAGE#######################################################
            
            #The ValidMinLimit will be loaded as the TrainMax limit
            validMinLimit=trainMaxLimit+1

            #The ValidMaxLimit will be loaded as the interval after the begin + train size +validation size
            #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date  
            validMaxLimit=None
            while(validMaxLimit is None):
                try:
                    validMaxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

            ########################################TESTING STAGE######################################################## 
            #The TestMinLimit will be loaded as the ValidMaxlimit 
            testMinLimit=validMaxLimit+1

            #The testMaxLimit will be loaded as the interval after the begin + train size +validation size + Testsize
            #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date 
            testMaxLimit=None
            while(testMaxLimit is None):
                try:
                    testMaxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize+self.testSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

            #Separate the Validation and testing data according to the limits found before
            #Prepare the training and validation files for saving them later 
            ensambleValid=pd.DataFrame(index=self.dates[validMinLimit:validMaxLimit].ix[:,'Date'].drop_duplicates().tolist())
            ensambleTest=pd.DataFrame(index=self.dates[testMinLimit:testMaxLimit].ix[:,'Date'].drop_duplicates().tolist())
            
            #Put the name of the index for validation and testing
            ensambleValid.index.name='Date'
            ensambleTest.index.name='Date'
            
           
            #Explorations are epochs, 
            for eps in self.explorations:

                #policy will be 0.2, so the randomness of predictions (actions) will happen with 20% of probability 
                self.policy.eps = eps[0]
                
                #there will be 100 iterations, or eps[1])
                for i in range(0,eps[1]):
                    
                    del(trainEnv)

                    #Define the training, validation and testing environments with their respective callbacks
                    trainEnv = SpEnv(operationCost=self.operationCost,minLimit=trainMinLimit,maxLimit=trainMaxLimit,callback=self.trainer)
                    del(validEnv)
                    validEnv=SpEnv(operationCost=self.operationCost,minLimit=validMinLimit,maxLimit=validMaxLimit,callback=self.validator,ensamble=ensambleValid,columnName="iteration"+str(i))
                    del(testEnv)
                    testEnv=SpEnv(operationCost=self.operationCost,minLimit=testMinLimit,maxLimit=testMaxLimit,callback=self.tester,ensamble=ensambleTest,columnName="iteration"+str(i))

                    #Reset the callback
                    self.trainer.reset()
                    self.validator.reset()
                    self.tester.reset()

                    #Reset the training environment
                    trainEnv.resetEnv()
                    #Train the agent
                    self.agent.fit(trainEnv,nb_steps=floor(self.trainSize.days-self.trainSize.days*0.2),visualize=False,verbose=0)
                    #Get the info from the train callback
                    (_,trainCoverage,trainAccuracy,trainReward,trainLongPerc,trainShortPerc,trainLongAcc,trainShortAcc,trainLongPrec,trainShortPrec)=self.trainer.getInfo()
                    #Print Callback values on the screen
                    print(str(i) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward))

                    #Reset the validation environment
                    validEnv.resetEnv()
                    #Test the agent on validation data
                    self.agent.test(validEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                    #Get the info from the validation callback
                    (_,validCoverage,validAccuracy,validReward,validLongPerc,validShortPerc,validLongAcc,validShortAcc,validLongPrec,validShortPrec)=self.validator.getInfo()
                    #Print callback values on the screen
                    print(str(i) + " VALID:  acc: " + str(validAccuracy)+ " cov: " + str(validCoverage)+ " rew: " + str(validReward))

                    #Reset the testing environment
                    testEnv.resetEnv()
                    #Test the agent on testing data
                    self.agent.test(testEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                    #Get the info from the testing callback
                    (_,testCoverage,testAccuracy,testReward,testLongPerc,testShortPerc,testLongAcc,testShortAcc,testLongPrec,testShortPrec)=self.tester.getInfo()
                    #Print callback values on the screen
                    print(str(i) + " TEST:  acc: " + str(testAccuracy)+ " cov: " + str(testCoverage)+ " rew: " + str(testReward))
                    print(" ")
                    
                    #write the walk data on the text file
                    self.outputFile[iteration].write(
                        str(i)+","+
                        str(trainAccuracy)+","+
                        str(trainCoverage)+","+
                        str(trainReward)+","+
                        str(trainLongPerc)+","+
                        str(trainShortPerc)+","+
                        str(trainLongAcc)+","+
                        str(trainShortAcc)+","+
                        str(trainLongPrec)+","+
                        str(trainShortPrec)+","+
                        
                        str(validAccuracy)+","+
                        str(validCoverage)+","+
                        str(validReward)+","+
                        str(validLongPerc)+","+
                        str(validShortPerc)+","+
                        str(validLongAcc)+","+
                        str(validShortAcc)+","+
                        str(validLongPrec)+","+
                        str(validShortPrec)+","+
                        
                        str(testAccuracy)+","+
                        str(testCoverage)+","+
                        str(testReward)+","+
                        str(testLongPerc)+","+
                        str(testShortPerc)+","+
                        str(testLongAcc)+","+
                        str(testShortAcc)+","+
                        str(testLongPrec)+","+
                        str(testShortPrec)+"\n")

            #Close the file                
            self.outputFile[iteration].close()

            #For the next walk, the current starting point will be the current starting point + the test size
            #It means that, for the next walk, the training data will start 6 months after the training data of 
            #the previous walk   
            self.currentStartingPoint+=self.testSize

            #Write validation and Testing Data into files
            #Save the files for processing later with the ensemble
            ensambleValid.to_csv("./Output/ensemble/walk"+str(iteration)+"ensemble_valid.csv")
            ensambleTest.to_csv("./Output/ensemble/walk"+str(iteration)+"ensemble_test.csv")

    #Function to end the Agent
    def end(self):
        import os 

        #Close the files where the results were written 
        for outputFile in self.outputFile:
            outputFile.close()


