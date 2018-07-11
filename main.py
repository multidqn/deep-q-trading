import IntradayPolicy
import SpEnv
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

environment = SpEnv.getEnv()
nb_actions = environment.action_space.n
print(environment.observation_space.shape)

model = Sequential()
model.add(Flatten(input_shape=(6,) + environment.observation_space.shape))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


policy = IntradayPolicy.getPolicy(env = environment)
memory = SequentialMemory(limit=100000, window_length=6)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(environment, nb_steps=20000, visualize=False, verbose=1)