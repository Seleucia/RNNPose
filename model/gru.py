import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_bias
from helper.optimizer import RMSprop

dtype = T.config.floatX

class gru:
   def __init__(self, inputSize, hiddenSize, name = "GRU Layer", E = None, U = None, W = None, b = None, moduleType = "GRU"):

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.name = name
        self.moduleType = moduleType

        if E == None:
            E = np.random.uniform(-np.sqrt(1./inputSize), np.sqrt(1./inputSize), (hiddenSize, inputSize))
            U = np.random.uniform(-np.sqrt(1./hiddenSize), np.sqrt(1./hiddenSize), (3 * 1, hiddenSize, hiddenSize))
            W = np.random.uniform(-np.sqrt(1./hiddenSize), np.sqrt(1./hiddenSize), (3 * 1, hiddenSize, hiddenSize))
            b = np.zeros((3 * 1, hiddenSize))

        self.E = theano.shared(name = name + '.E', value = E.astype(theano.config.floatX))
        self.U = theano.shared(name = name + '.U', value = U.astype(theano.config.floatX))
        self.W = theano.shared(name = name + '.W', value = W.astype(theano.config.floatX))
        self.b = theano.shared(name = name + '.b', value = b.astype(theano.config.floatX))

    def step(self, x, s_prev):

        x_emb = self.E[:, x]

        z = T.nnet.hard_sigmoid(self.W[0].dot(x_emb) + self.U[0].dot(s_prev) + self.b[0])
        r = T.nnet.hard_sigmoid(self.W[1].dot(x_emb) + self.U[1].dot(s_prev) + self.b[1])
        h = T.tanh(self.W[2].dot(x_emb) + self.U[2].dot(s_prev * r) + self.b[2])
        s = (T.ones_like(z) - z) * h + z * s_prev

        return s