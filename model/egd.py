import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_bias,get_err_fn
from helper.optimizer import RMSprop,Adam
import helper.utils as u
dtype = T.config.floatX


class egd:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, single_output=True, output_activation=theano.tensor.nnet.relu,cost_function='mse',optimizer = RMSprop):

       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       self.n_fc1=256
       self.n_fc2=256
       self.n_fc3=256


       self.W_fc1 = init_weight((self.n_fc1, self.n_fc2),'W_fc1', 'glorot')
       self.b_fc1 = init_bias(self.n_fc2, sample='zero')

       self.W_fc2 = init_weight((self.n_fc2, self.n_fc3),'W_fc2', 'glorot')
       self.b_fc2 =init_bias(self.n_fc3, sample='zero')

       self.W_fc3 = init_weight((self.n_fc3, self.n_out),'w_fc3', 'glorot')
       self.b_fc3 =init_bias(self.n_out, sample='zero')

       self.W_xr = init_weight((self.n_in, self.n_lstm), 'W_xr', 'glorot')
       self.W_hr = init_weight((self.n_lstm, self.n_lstm), 'W_hr', 'ortho')
       self.b_r  = init_bias(self.n_lstm, sample='zero')
       self.W_xz = init_weight((self.n_in, self.n_lstm), 'W_xz', 'glorot')
       self.W_hz = init_weight((self.n_lstm, self.n_lstm), 'W_hz', 'ortho')
       self.b_z = init_bias(self.n_lstm, sample='zero')
       self.W_xh = init_weight((self.n_in, self.n_lstm), 'W_xh', 'glorot')
       self.W_hh = init_weight((self.n_lstm, self.n_lstm), 'W_hh', 'ortho')
       self.b_h = init_bias(self.n_lstm, sample='zero')
       self.W_hy = init_weight((self.n_lstm, self.n_fc1),'W_hy', 'glorot')
       self.b_y = init_bias(self.n_fc1, sample='zero')

       self.params = [self.W_xr, self.W_hr, self.b_r,
                      self.W_xz, self.W_hz, self.b_z,
                      self.W_xh, self.W_hh, self.b_h,
                      self.W_hy, self.b_y,self.W_fc1, self.b_fc1,self.W_fc2, self.b_fc2,self.W_fc3, self.b_fc3]

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


       #Hidden layer
       fc1_out = T.tanh(T.dot(y_vals, self.W_fc1)  + self.b_fc1)
       fc2_out = T.tanh(T.dot(fc1_out, self.W_fc2)  + self.b_fc2)
       fc3_out = T.tanh(T.dot(fc2_out, self.W_fc3)  + self.b_fc3)

       self.output=fc3_out.dimshuffle(1,0,2)

       cost=get_err_fn(self,cost_function,Y)
       _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )

       self.train = theano.function(inputs=[X, Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X], outputs = self.output,allow_input_downcast=True)
       self.n_param=n_lstm*n_lstm*4+n_in*n_lstm*4+n_lstm*n_out+n_lstm*3
