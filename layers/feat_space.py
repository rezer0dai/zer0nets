import numpy as np

import abc

# TODO : rethink benefit of bias term
class FeatureSpace(object):#, metaclass=abc.ABCMeta):
    def __init__(self, degree, connected_degree, index = -1, update_rate = 1.):
        self.b = np.zeros((1, connected_degree))
        self.w = np.random.normal(.0, degree**-.5, (degree, connected_degree))

        self.out = None
        self.error = None

        self.delta = []
        self.bias = []

        self.index = index
        self.update_rate = update_rate

    def fire_signal(self, inputs):
        assert inputs.shape[1] == self.w.shape[0], "forward pass fail with '{}' stacked at {}".format(self.name(), self.index)

        # transformation of input to connected space
        transformed_space__in = inputs.dot(self.w) + self.b
        # applying non linearity, will server as input to connected space
        #  import math
        #  if math.isnan(transformed_space__in[0][0]):
        #      print(self.name(), self.w, self.b)
        self.out = self.signal(transformed_space__in)
        #  if self.out.shape[0] == self.out.shape[1] and self.out.shape[0] == 1: print(transformed_space__in)
        return self.out

    def eval_signal(self, error, inputs):
        assert inputs.shape[1] == self.w.shape[0], "backward pass fail with '{}' stacked at {}".format(self.name(), self.index)
        ## Chain-Rule for derivatives ##
        error_prime = error * self.prime(self.signal_out())
        # propagating gradient on our weights to contribute to gradient
        self.error = error_prime.dot(self.w.T)

        # f' * x to compute our gradient
        self.delta.append(inputs.T.dot(error_prime) / inputs.shape[0])
        # for bias just collect all partial derivatives
        self.bias.append(np.sum(error_prime, axis = 0, keepdims = True) / inputs.shape[0])
        return self.error

    def update(self, learning_rate):
        assert len(self.delta) > 0, "tring to update layer without back-propagation ?!"

        self.w += self.update_rate * learning_rate * (sum(self.delta) / len(self.delta))
        self.b += self.update_rate * learning_rate * (sum(self.bias) / len(self.bias))


## TESING avoiding inf and nan in weiights ...
        #  self.w = np.reshape(np.clip(self.w, -500, 500), self.w.shape)
        #  self.b = np.reshape(np.clip(self.b, -500, 500), self.b.shape)

        #  dx = np.ones_like(self.w)
        #  dx[abs(self.w) < 1e-50] = 0
        #  self.w = self.w * dx
## TESING avoiding inf and nan in weiights ...

        self.delta = []
        self.bias = []

    def calc_err(self, y_predicted, y):
        self.error = self.__calc_err__(y_predicted, y)
        return self.error

    def signal_out(self):
        return self.out

#    @abc.abstractmethod
    def name(self):
        pass
#    @abc.abstractmethod
    def signal(self, x):
        pass
#    @abc.abstractmethod
    def prime(self, x):
        pass
