from rl.policy import Policy
import spEnv
import numpy

class IntradayPolicy(Policy):
    def __init__(self, env, eps=.1, stopLoss = 0, minOperationLength = 0):
        super(IntradayPolicy, self).__init__()
        self.env = env
        self.eps = eps
        self.stopLoss = stopLoss
        self.prevState = env.getCurrentState()
        self.minOperationLength = minOperationLength
        self.waitSteps = 0
        
    def select_action(self, q_values):
        if self.prevState != self.env.getCurrentState():
            self.waitSteps = self.minOperationLength
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if numpy.random.uniform() < self.eps:
            action = numpy.random.random_integers(0, nb_actions-1)
            if action != 0 and action == self.env.getCurrentState():
                    prevAction = action
                    while action == prevAction:
                        action = numpy.random.random_integers(0, nb_actions-1)
        else:
            action = numpy.argmax(q_values)
            if action != 0 and action == self.env.getCurrentState():
                if action == 1:
                    if numpy.argmin(q_values) == 0:
                        action = 2
                    else:
                        action = 0
                else:
                    if numpy.argmin(q_values) == 0:
                        action = 1
                    else:
                        action = 0
        
        today, tomorrow = self.env.getTodayTomorrow()

        if today != tomorrow:
            if self.env.getCurrentState() == 0:
                action = 0
            elif self.env.getCurrentState() == 1:
                action = 2
            else:
                action = 1

        if self.waitSteps > 0:
            #print("WAITING STEPS " + str(self.waitSteps))
            action = 0
            self.waitSteps -= 1

        if (self.stopLoss != 0) and (self.env.getProfit() <= self.stopLoss):
            #print("STOPPING LOSS")
            if self.env.getCurrentState() == 1:
                action = 2
            elif self.env.getCurrentState() == 2:
                action = 1

        

        self.prevState = self.env.getCurrentState()

        if self.env.getCurrentState() == 0:
            #print("HOLDING")
            self.waitSteps = 0
        
        #print("State: " + str(self.env.getCurrentState()) + "Action: " + str(action))
        
        if self.env.getCurrentState() == 3:
            action = 0
        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy
        # Returns
            Dict of config
        """
        config = super(IntradayPolicy, self).get_config()
        config['eps'] = self.eps
        return config

    def set_eps(self, eps):
        self.eps = eps

def getPolicy(env, eps=0.1, stopLoss = 0, minOperationLength = 0):
    return IntradayPolicy(env, eps, stopLoss, minOperationLength)