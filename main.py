"""import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";"""

from DeepQTradingWV import DeepQTrading
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import sys
import telegram


bot = telegram.Bot(token='864997856:AAFjYS9qw9Gd3L_AwYGBPdhE7w-SWxf-JjU')

startingTime=datetime.datetime.now()

bot.send_message(chat_id='@DeepQTrading', text="Esperimento Iniziato "+str(datetime.datetime.now()))

nb_actions = 3

model = Sequential()
model.add(Flatten(input_shape=(50,1,68)))
model.add(Dense(256,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(512,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(256,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(nb_actions))
model.add(Activation('linear'))





dqt = DeepQTrading(
    model=model,
    explorations=[(0.1,50)],
    trainSize=datetime.timedelta(days=360*10),
    validationSize=datetime.timedelta(days=30*6),
    testSize=datetime.timedelta(days=30*6),
    outputFile="walks",
    begin=datetime.datetime(2004,1,1,0,0,0,0),
    end=datetime.datetime(2018,2,28,0,0,0,0),
    nbActions=nb_actions,
    nOutput=15
    )


#TODO: 2 walk di cui stamparne ogni singola iterazione di dimensione 10 anni, 1 anno, 1 anno

try:
    dqt.run()
    bot.send_message(chat_id='@DeepQTrading', text="Finito senza errori -- "+str(datetime.datetime.now()))
except: 
    bot.send_message(chat_id='@DeepQTrading', text="Finito con errori -- "+str(datetime.datetime.now()))

bot.send_message(chat_id='@DeepQTrading', text="Ho impiegato messo"+str(datetime.datetime.now() - startingTime))

dqt.end()