import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_pweight,init_bias
from helper.optimizer import RMSprop

dtype = T.config.floatX

class gru:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, output_activation=theano.tensor.nnet.relu,cost_function='nll',optimizer = RMSprop):
       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       self.W_xr = init_weight((self.n_in, self.n_lstm), 'W_xr', 'glorot')
       self.W_hr = init_weight((self.n_lstm, self.n_lstm), 'W_hr', 'ortho')
       self.b_r  = init_bias(self.n_lstm, sample='zero')
       self.W_xz = init_weight((self.n_in, self.n_lstm), 'W_xz', 'glorot')
       self.W_hz = init_weight((self.n_lstm, self.n_lstm), 'W_hz', 'ortho')
       self.b_z = init_bias(self.n_lstm, sample='zero')
       self.W_xh = init_weight((self.n_in, self.n_lstm), 'W_xh', 'glorot')
       self.W_hh = init_weight((self.n_lstm, self.n_lstm), 'W_hh', 'ortho')
       self.b_h = init_bias(self.n_lstm, sample='zero')
       self.W_hy = init_weight((self.n_lstm, self.n_out),'W_hy', 'glorot')
       self.b_y = init_bias(self.n_out, sample='zero')
       self.one_mat=T.ones((batch_size,n_lstm),dtype=dtype)

       self.params = [self.W_xr, self.W_hr, self.b_r,
                      self.W_xz, self.W_hz, self.b_z,
                      self.W_xh, self.W_hh, self.b_h,
                      self.W_hy, self.b_y]

       def step_lstm(x_t, h_tm1):
           r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tm1, self.W_hr) + self.b_r)
           z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tm1, self.W_hz)  + self.b_z)
           h_t = T.tanh(T.dot(x_t, self.W_xh) + T.dot((r_t*h_tm1),self.W_hh)  + self.b_h)
           hh_t = z_t * h_tm1 + (1-z_t)*h_t
           #y_t = T.nnet.sigmoid(T.dot(hh_t, self.W_hy) + self.b_y)
           y_t = T.tanh(T.dot(hh_t, self.W_hy) + self.b_y)
           #y_t = T.dot(hh_t, self.W_hy) + self.b_y
           return [h_t, y_t]

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector
       #Y_NaN= T.tensor3() # batch of sequence of vector
       h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

       [h_vals, y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=X.dimshuffle(1,0,2),
                                         outputs_info=[h0, None])

       self.output = y_vals.dimshuffle(1,0,2)

       cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
       nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))
       #mse = T.mean((self.output - Y) ** 2)

       #c_ = switch(theano.tensor.eq(c, 0), VALUE_WHEN_0, c) # c with 0s replaced by VALUE_WHEN_0

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

       _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )


       self.train = theano.function(inputs=[X, Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
       self.n_param=n_lstm*n_lstm*3+n_in*n_lstm*3+n_lstm*n_out+n_lstm*3
