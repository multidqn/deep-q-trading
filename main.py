import IntradayPolicy
import SpEnv
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import datetime


#842063
environment = SpEnv.getEnv(maxLimit = 842063)
testEnv = SpEnv.getEnv(minLimit = 842063, verbose=True, operationCost = 1)
nb_actions = environment.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(300,) + environment.observation_space.shape))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))


policy = IntradayPolicy.getPolicy(env = environment, eps = 0.5, stopLoss=-500, minOperationLength=5)
policyTest = IntradayPolicy.getPolicy(env = testEnv, eps = 0, stopLoss=-500, minOperationLength=5)
memory = SequentialMemory(limit=100000, window_length=300)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
target_model_update=1e-2, policy=policy, test_policy=policyTest)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.load_weights("Q.weights")

print(datetime.datetime.now())


for i in range(12):
    
    print(i)
    
    policy.set_eps(0.5)
    dqn.fit(environment, nb_steps=2000, visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)


    policy.set_eps(0.1)
    dqn.fit(environment, nb_steps=20000, visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)

    policy.set_eps(0.07)
    dqn.fit(environment, nb_steps=200000, visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)
    
    policy.set_eps(0.007)
    dqn.fit(environment, nb_steps=200000, visualize=False, verbose=0)
    dqn.save_weights("Q.weights", overwrite=True)

print("End of traning")
print(datetime.datetime.now())
dqn.test(testEnv, nb_episodes=671, verbose=0, visualize=False)
