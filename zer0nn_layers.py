from layers.input import *
from layers.linear import *
from layers.sigmoid import Sigmoid
from layers.relu import *
from layers.experimental import *

class SpaceLayerFactory:
    def layer(self, name, dim1, dim2, ind):
        if "input" == name:
            assert False, "input layer must be add-hot pluged to neural net at forward pass per new input!"
        if "lin" == name:
            return Linear(dim1, dim2, ind)
        if "sig" == name:
            return Sigmoid(dim1, dim2, ind)
        if "relu" == name:
            return ReLU(dim1, dim2, ind)
        if "lrelu" == name:
            return LeakyReLU(dim1, dim2, ind)

        if "pgd" == name:
            return PolicyGradient(dim1, dim2, ind)
        if "tan" == name:
            return TanH(dim1, dim2, ind)
        if "linc" == name:
            return LinearCrossEntropy(dim1, dim2, ind)

        if "nes" == name:
            return EvolutionarySearch(dim1, dim2, ind)
        if "softmax" == name:
            return SoftMax(dim1, dim2, ind)

        assert False, "space-layer : <{}> not implemented!".format(name)
        return None
