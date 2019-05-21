from SpEnv import SpEnv
from Callback import ValidationCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from math import floor
import pandas as pd
import datetime
import telegram

MK="sp500"

class DeepQTrading:
    def __init__(self, model, explorations, trainSize, validationSize, testSize, outputFile, begin, end, nbActions, nOutput=1, operationCost=0,telegramToken="",telegramChatID=""):
        if(telegramToken!="" and telegramChatID!=""):
            self.chatID=telegramChatID
            self.telegramOutput=True
            try:
                self.bot = telegram.Bot(token=telegramToken)
        else:
            self.telegramOutput=True

        self.policy = EpsGreedyQPolicy()
        self.explorations=explorations
        self.nbActions=nbActions
        self.model=model
        self.memory = SequentialMemory(limit=10000, window_length=1)
        self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                    enable_double_dqn=True,enable_dueling_network=True)
        self.agent.compile(Adam(lr=1e-3), metrics=['mae'])
        self.agent.save_weights("q.weights", overwrite=True)
        self.currentStartingPoint = begin
        self.trainSize=trainSize
        self.validationSize=validationSize
        self.testSize=testSize
        self.walkSize=trainSize+validationSize+testSize
        self.endingPoint=end


        self.dates= pd.read_csv('./dataset/'+MK+'Hour.csv')

        self.sp = pd.read_csv('./dataset/'+MK+'Hour.csv')
        self.sp['Datetime'] = pd.to_datetime(self.sp['Date'] + ' ' + self.sp['Time'])
        self.sp = self.sp.set_index('Datetime')
        self.sp = self.sp.drop(['Time','Date'], axis=1)
        self.sp = self.sp.index


        self.operationCost = operationCost
        self.trainer=ValidationCallback()
        self.validator=ValidationCallback()
        self.tester=ValidationCallback()
        self.outputFile=[]

        for i in range(0,nOutput):
            self.outputFile.append(open(outputFile+str(i+1)+".csv", "w+"))
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
        trainEnv=validEnv=testEnv=" "

        iteration=-1

        while(self.currentStartingPoint+self.walkSize <= self.endingPoint):


            iteration+=1
            if(self.telegramOutput):
                self.bot.send_message(chat_id=self.chatID, text="Iteration "+str(iteration + 1 )+" started.")
            
            
            del(self.memory)
            del(self.agent)
            self.memory = SequentialMemory(limit=10000, window_length=1)
            self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                    enable_double_dqn=True,enable_dueling_network=True)
            self.agent.compile(Adam(lr=1e-3), metrics=['mae'])
            self.agent.load_weights("q.weights")

            trainMinLimit=None

            while(trainMinLimit is None):
                try:
                    trainMinLimit = self.sp.get_loc(self.currentStartingPoint)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)
            trainMaxLimit=None

            while(trainMaxLimit is None):
                try:
                    trainMaxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

            validMinLimit=trainMaxLimit


            validMaxLimit=None
            while(validMaxLimit is None):
                try:
                    validMaxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)


            testMinLimit=validMaxLimit


            testMaxLimit=None
            while(testMaxLimit is None):
                try:
                    testMaxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize+self.testSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

            
            ensambleValid=pd.DataFrame(index=self.dates[validMinLimit:validMaxLimit].ix[:,'Date'].drop_duplicates().tolist())
            ensambleTest=pd.DataFrame(index=self.dates[testMinLimit:testMaxLimit].ix[:,'Date'].drop_duplicates().tolist())
            ensambleValid.index.name='Date'
            ensambleTest.index.name='Date'
            for eps in self.explorations:
                self.policy.eps = eps[0]
                
                for i in range(0,eps[1]):
                    del(trainEnv)
                    trainEnv = SpEnv(operationCost=self.operationCost,minLimit=trainMinLimit,maxLimit=trainMaxLimit,callback=self.trainer)
                    del(validEnv)
                    validEnv=SpEnv(operationCost=self.operationCost,minLimit=validMinLimit,maxLimit=validMaxLimit,callback=self.validator,ensamble=ensambleValid,columnName="iteration"+str(i))
                    del(testEnv)
                    testEnv=SpEnv(operationCost=self.operationCost,minLimit=testMinLimit,maxLimit=testMaxLimit,callback=self.tester,ensamble=ensambleTest,columnName="iteration"+str(i))

                    self.trainer.reset()
                    self.validator.reset()
                    self.tester.reset()

                    trainEnv.resetEnv()
                    self.agent.fit(trainEnv,nb_steps=floor(self.trainSize.days-self.trainSize.days*0.2),visualize=False,verbose=0)
                    (_,trainCoverage,trainAccuracy,trainReward,trainLongPerc,trainShortPerc,trainLongAcc,trainShortAcc,trainLongPrec,trainShortPrec)=self.trainer.getInfo()
                    print(str(i) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward))

                    validEnv.resetEnv()
                    self.agent.test(validEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                    (_,validCoverage,validAccuracy,validReward,validLongPerc,validShortPerc,validLongAcc,validShortAcc,validLongPrec,validShortPrec)=self.validator.getInfo()
                    print(str(i) + " VALID:  acc: " + str(validAccuracy)+ " cov: " + str(validCoverage)+ " rew: " + str(validReward))

                    testEnv.resetEnv()
                    self.agent.test(testEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                    (_,testCoverage,testAccuracy,testReward,testLongPerc,testShortPerc,testLongAcc,testShortAcc,testLongPrec,testShortPrec)=self.tester.getInfo()
                    print(str(i) + " TEST:  acc: " + str(testAccuracy)+ " cov: " + str(testCoverage)+ " rew: " + str(testReward))

                    print(" ")

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
            self.outputFile[iteration].close()
            self.currentStartingPoint+=self.testSize

            ensambleValid.to_csv("./Output/ensamble/walk"+str(iteration)+"ensamble_valid.csv")
            
            ensambleTest.to_csv("./Output/ensamble/walk"+str(iteration)+"ensamble_test.csv")

    def end(self):
        import os 
        for outputFile in self.outputFile:
            outputFile.close() 
        os.remove("q.weights")

