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

class autoencoder(object):
    def __init__(self,rng,params,cost_function='mse',optimizer = RMSprop):

        lr=params["lr"]
        batch_size=params["batch_size"]
        n_output=params['n_output']
        corruption_level=params["corruption_level"]

        X = T.matrix(name="input",dtype=dtype) # batch of sequence of vector
        Y = T.matrix(name="output",dtype=dtype) # batch of sequence of vector
        is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction
        bin_noise=rng.binomial(size=(batch_size,n_output/3,1), n=1,p=1 - corruption_level,dtype=theano.config.floatX)
        #bin_noise_3d= T.reshape(T.concatenate((bin_noise, bin_noise,bin_noise),axis=1),(batch_size,n_output/3,3))
        bin_noise_3d= T.concatenate((bin_noise, bin_noise,bin_noise),axis=2)

        noise= rng.normal(size=(batch_size,n_output), std=0.03, avg=0.0,dtype=theano.config.floatX)
        noise_bin=T.reshape(noise,(batch_size,n_output/3,3))*bin_noise_3d
        X_train=T.reshape(noise_bin,(batch_size,n_output))+X

        X_tilde= T.switch(T.neq(is_train, 0), X_train, X)

        W_1_e =u.init_weight(shape=(n_output,1024),rng=rng,name="w_hid",sample="glorot")
        b_1_e=u.init_bias(1024,rng)

        W_2_e =u.init_weight(shape=(1024,2048),rng=rng,name="w_hid",sample="glorot")
        b_2_e=u.init_bias(2048,rng)

        W_2_d = W_2_e.T
        b_2_d=u.init_bias(1024,rng)

        W_1_d = W_1_e.T
        b_1_d=u.init_bias(n_output,rng)

        h_1_e=HiddenLayer(rng,X_tilde,0,0, W=W_1_e,b=b_1_e,activation=nn.relu)
        h_2_e=HiddenLayer(rng,h_1_e.output,0,0, W=W_2_e,b=b_2_e,activation=nn.relu)
        h_2_d=HiddenLayer(rng,h_2_e.output,0,0, W=W_2_d,b=b_2_d,activation=u.do_nothing)
        h_1_d=LogisticRegression(rng,h_2_d.output,0,0, W=W_1_d,b=b_1_d)

        self.output = h_1_d.y_pred

        self.params =h_1_e.params+h_2_e.params
        self.params.append(b_2_d)
        self.params.append(b_1_d)

        cost=get_err_fn(self,cost_function,Y)
        L2_reg=0.0001
        L2_sqr = theano.shared(0.)
        for param in self.params:
            L2_sqr += (T.sum(param[0] ** 2)+T.sum(param[1] ** 2))

        cost += L2_reg*L2_sqr

        _optimizer = optimizer(cost, self.params, lr=lr)
        self.train = theano.function(inputs=[X,Y,is_train],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
        self.predictions = theano.function(inputs = [X,is_train], outputs = self.output,allow_input_downcast=True)
        self.mid_layer = theano.function(inputs = [X,is_train], outputs = h_2_e.output,allow_input_downcast=True)
        self.n_param=count_params(self.params)