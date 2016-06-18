from theano import shared
import helper.utils as u
import numpy as np
import theano
import theano.tensor as T
from layers import ConvLayer,PoolLayer,HiddenLayer,DropoutLayer, LogisticRegression
import theano.tensor.nnet as nn
from helper.utils import init_weight,init_bias,get_err_fn,count_params, do_nothing
from helper.optimizer import RMSprop

# theano.config.exception_verbosity="high"
dtype = T.config.floatX

class cnn(object):
    def __init__(self,rng,params,cost_function='mse',optimizer = RMSprop):

        lr=params["lr"]
        batch_size=params["batch_size"]
        n_output=params['n_output']
        corruption_level=params["corruption_level"]

        # minibatch)
        X = T.matrix(name="input",dtype=dtype) # batch of sequence of vector
        Y = T.matrix(name="output",dtype=dtype) # batch of sequence of vector
        shape=(batch_size,n_output)
        X_tilde= self.theano_rng.binomial(size=X.shape, n=1, p=1 - corruption_level,
                                        dtype=theano.config.floatX) * X

        W_1_e =u.init_weight(shape=(96,1024),rng=rng,name="w_hid",sample="glorot")
        b_1_e=u.init_bias(1024,rng)

        W_2_e =u.init_weight(shape=(1024,2048),rng=rng,name="w_hid",sample="glorot")
        b_2_e=u.init_bias(2048,rng)

        W_2_d = W_2_e.T
        b_2_d=u.init_bias(1024,rng)

        W_1_d = W_1_e.T
        b_1_d=u.init_bias(96,rng)


        h_1_e=HiddenLayer(rng,X_tilde,0,0, W=W_1_e,b=b_1_e,activation=nn.relu)
        h_2_e=HiddenLayer(rng,h_1_e.output,0,0, W=W_2_e,b=b_2_e,activation=nn.relu)
        h_2_d=HiddenLayer(rng,h_2_e.output,0,0, W=W_2_d,b=b_2_d,activation=nn.relu)
        h_1_d=HiddenLayer(rng,h_2_d.output,0,0, W=W_1_d,b=b_1_d,activation=nn.relu)

        self.output = h_1_e.output

        self.params =h_1_e.params+h_2_e.params+h_2_d.params+h_1_d.params

        cost=get_err_fn(self,cost_function,Y)
        L2_reg=0.0001
        L2_sqr = theano.shared(0.)
        for param in self.params:
            L2_sqr += (T.sum(param[0] ** 2)+T.sum(param[1] ** 2))

        cost += L2_reg*L2_sqr

        _optimizer = optimizer(cost, self.params, lr=lr)
        self.train = theano.function(inputs=[X,Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
        self.predictions = theano.function(inputs = [X], outputs = self.output,allow_input_downcast=True)
        self.n_param=count_params(self.params)