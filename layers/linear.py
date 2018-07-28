import numpy as np
from feat_space import *

class Linear(FeatureSpace):
    def name(self):
        return "linear"
    def signal(self, x):
        return x
    def prime(self, _):
        return 1.
