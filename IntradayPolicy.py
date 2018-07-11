from rl.policy import Policy
import SpEnv
import numpy

class IntradayPolicy(Policy):
    def __init__(self, env, eps=.1):
        super(IntradayPolicy, self).__init__()
        self.env = env
        self.eps = eps

    def select_action(self, q_values):
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
        
        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy
        # Returns
            Dict of config
        """
        config = super(IntradayPolicy, self).get_config()
        config['eps'] = self.eps
        return config

def getPolicy(env):
    return IntradayPolicy(env)