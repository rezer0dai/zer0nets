import numpy as np
from feat_space import *

# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
# https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
class SoftMax(FeatureSpace):
    def signal(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
    def prime(self, x):
        return x - 1.# should only decrease true label ...
        s = x.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    def name(self):
        return "softmax"

class TanH(FeatureSpace):
    def signal(self, x):
        return 1. * np.tanh(x)
    def prime(self, x):
        return 1. * (1 - np.tanh(x) ** 2)
    def name(self):
        return "tanh"

# we are hardcoding in error function sigma to .1 therefore reflect it here as well ; later build rs framework with this in mind
class EvolutionarySearch(FeatureSpace):
    def name(self):
        return "evolution layer"

    def signal(self, x):
        return x + .1 * np.random.randn(x.shape[0], x.shape[1]) # add noise
    def prime(self, x):
        return 1.1
