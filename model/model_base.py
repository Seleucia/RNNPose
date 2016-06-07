from helper.optimizer import RMSprop
import numpy as np
import theano
import theano.tensor as T
from theano import shared

class model_base():
    def __init__(self):
        print "Base created"

    def get_err_fn(self,cost_function,Y):
       cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
       nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))

       tmp = self.output - Y
       tmp = theano.tensor.switch(theano.tensor.isnan(tmp),0,tmp)
       mse = T.mean((tmp) ** 2)

       cost = 0
       if cost_function == 'mse':
           cost = mse
       elif cost_function == 'cxe':
           cost = cxe
       else:
           cost = nll




