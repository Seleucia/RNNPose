import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from helper.utils import init_weight,init_bias,get_err_fn
from helper.optimizer import RMSprop

dtype = T.config.floatX

class gru():
   def __init__(self,rng, n_in, n_lstm, n_out, lr=0.00001, batch_size=64, output_activation=theano.tensor.nnet.relu,cost_function='mse',optimizer = RMSprop):
       # rng = RandomStreams(seed=1234)
       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       self.W_xr = init_weight((self.n_in, self.n_lstm),rng=rng,name='W_xi',sample= 'glorot')
       self.W_hr = init_weight((self.n_lstm, self.n_lstm),rng=rng,  name='W_hr', sample='glorot')
       self.b_r  = init_bias(self.n_lstm,rng=rng, sample='zero')
       self.W_xz = init_weight((self.n_in, self.n_lstm), rng=rng, name='W_xz', sample='glorot')
       self.W_hz = init_weight((self.n_lstm, self.n_lstm), rng=rng, name='W_hz', sample='glorot')
       self.b_z = init_bias(self.n_lstm,rng=rng, sample='zero')
       self.W_xh = init_weight((self.n_in, self.n_lstm), rng=rng, name='W_xh', sample='glorot')
       self.W_hh = init_weight((self.n_lstm, self.n_lstm), rng=rng, name='W_hh',sample= 'glorot')
       self.b_h = init_bias(self.n_lstm,rng=rng, sample='zero')
       self.W_hy = init_weight((self.n_lstm, self.n_out), rng=rng,name='W_hy', sample= 'glorot')
       self.b_y = init_bias(self.n_out,rng=rng, sample='zero')
       self.one_mat=T.ones((batch_size,n_lstm),dtype=dtype)

       self.params = [self.W_xr, self.W_hr, self.b_r,
                      self.W_xz, self.W_hz, self.b_z,
                      self.W_xh, self.W_hh, self.b_h,
                      self.W_hy, self.b_y]

       def step_lstm(x_t, h_tm1):
           r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tm1, self.W_hr) + self.b_r)
           z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tm1, self.W_hz)  + self.b_z)
           h_t = T.tanh(T.dot(x_t, self.W_xh) + T.dot((r_t*h_tm1),self.W_hh)  + self.b_h)
           hh_t = z_t * h_t + (1-z_t)*h_tm1
           y_t = T.tanh(T.dot(hh_t, self.W_hy) + self.b_y)
           return [hh_t, y_t]



       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector
       #Y_NaN= T.tensor3() # batch of sequence of vector
       h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

       [h_vals, y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=X.dimshuffle(1,0,2),
                                         outputs_info=[h0, None])

       self.output = y_vals.dimshuffle(1,0,2)

       cost=get_err_fn(self,cost_function,Y)

       _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )


       self.train = theano.function(inputs=[X, Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
       self.n_param=n_lstm*n_lstm*3+n_in*n_lstm*3+n_lstm*n_out+n_lstm*3
