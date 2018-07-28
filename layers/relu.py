from feat_space import *

class ReLU(FeatureSpace):
    def name(self):
        return "ReLU"

    def signal(self, x):
        saturated_part = x > .0
        return saturated_part * x

    def prime(self, x):
        saturated_part = x > .0
        return saturated_part

class LeakyReLU(FeatureSpace):
    #  def __init__(self, degree, connected_degree, index = -1, update_rate = 1., alpha = .0001):
    #      super(FeatureSpace, self).__init__(degree, connected_degree, index, update_rate)
    #      self.alpha = alpha

    def name(self):
        return "LeakyReLU"

    def signal(self, x):
        saturated_part = x > .0
        leaky_part = x < .0
        return saturated_part * x + leaky_part * x * .0001#self.alpha

    def prime(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = .0001#self.alpha
        return dx
