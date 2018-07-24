import IntradayPolicy
import SpEnv
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.callbacks import FileLogger


#842063
environment = SpEnv.getEnv(maxLimit = 842063)
testEnv = SpEnv.getEnv(minLimit = 842063, verbose=True, operationCost = 1)
nb_actions = environment.action_space.n
#print(environment.observation_space.shape)

model = Sequential()
model.add(Flatten(input_shape=(100,) + environment.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
#print(model.summary())


policy = IntradayPolicy.getPolicy(env = environment, eps = 0.5, stopLoss=-500, minOperationLength=5)
policyTest = IntradayPolicy.getPolicy(env = testEnv, eps = 0, stopLoss=-500, minOperationLength=5)
memory = SequentialMemory(limit=100000, window_length=100)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
target_model_update=1e-2, policy=policy, test_policy=policyTest)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


dqn.load_weights("Q.weights")

for i in range(5):
    
    print(i)
    
    policy.set_eps(0.5)
    dqn.fit(environment, nb_steps=2000, callbacks=[FileLogger("Episodes.json", interval=100)], visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)


    policy.set_eps(0.1)
    dqn.fit(environment, nb_steps=20000, callbacks=[FileLogger("Episodes.json", interval=100)], visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)

    policy.set_eps(0.07)
    dqn.fit(environment, nb_steps=200000, callbacks=[FileLogger("Episodes.json", interval=100)], visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)
    
    policy.set_eps(0.007)
    dqn.fit(environment, nb_steps=200000, callbacks=[FileLogger("Episodes.json", interval=100)], visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)

dqn.test(testEnv, nb_episodes=671, verbose=0, visualize=False)
