from theano import shared
import numpy as np
import theano.tensor as T

dtype = T.config.floatX

class RMSprop:
    def __init__(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):

        self.cost = cost
        self.params = params
        self.lr = shared(np.cast[dtype](lr))
        self.rho = shared(np.cast[dtype](rho))
        self.epsilon = shared(np.cast[dtype](epsilon))
        self.gparams = T.grad(self.cost, self.params)

    def getUpdates(self):

        acc = [shared(np.zeros(p.get_value(borrow=True).shape, dtype=dtype)) for p in self.params]
        updates = []

        for p, g, a in zip(self.params, self.gparams, acc):
            new_a = self.rho * a + (1 - self.rho) * (g ** 2)
            updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))

        return updates