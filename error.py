import numpy as np

def mse(y_predicted, y):
    return -1. * ((y - y_predicted) ** 2)
def abs_e(y_predicted, y):
    return -1. * abs(y - y_predicted)
def cross_entropy(y_predicted, y):
    return -1. * (np.divide(y, y_predicted) - np.divide(1 - y, 1 - y_predicted))
def policy_gradient(y_predicted, y_r):
    return -1. * np.log((y_r[1] - y_predicted)**2 + 1e-4) * y_r[0]
#    return -1. * np.log(y_predicted**2 + 1e-10) * y_r

#  def abs_KL_div(y_true, y_pred):
#      y_true = K.clip(y_true, K.epsilon(), None)
#      y_pred = K.clip(y_pred, K.epsilon(), None)
#      return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)

# this seems hard to intercorporate as loss function ...
def evolution_search(y_predicted, rewards): # we are hardcoding here sigma to .1 ; later build rs framework with this in mind
    #OpenAI ~ standardize the rewards to have a gaussian distribution
    rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    #OpenAI ~ Karpathy ~ approapriate normalization
    return (1. / .1) * rewards

class LossFactory:
    def loss(self, name):
        if "mse" == name:
            return mse
        if "abs" == name:
            return abs_e
        if "ce" == name:
            return cross_entropy
        if "pgd" == name:
            return policy_gradient
        if "nes" == name:
            return evolution_search
        assert False, "unknown loss function {}".format(name)
        return None
