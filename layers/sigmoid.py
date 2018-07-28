import numpy as np
from feat_space import *

class Sigmoid(FeatureSpace):
    def name(self):
        return "sigmoid"
    def signal(self, x):
        dx = np.ones_like(x)
        dx[x < -700] = 0
        return 1 / (1 + np.exp(-(x * dx)))
    def prime(self, x):
        return x * (1. - x)
