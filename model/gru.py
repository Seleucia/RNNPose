import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_bias
from helper.optimizer import RMSprop
import helper.utils as u
dtype = T.config.floatX

class gru:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, single_output=True, output_activation=theano.tensor.nnet.relu,cost_function='nll',optimizer = RMSprop):
      prefix = "GRU_"
      self.n_in = n_in
      self.n_lstm = n_lstm
      self.n_out = n_out

      self.W_xr = init_weight((self.n_in, self.n_out), "W_xr" )
      self.W_hr = init_weight((self.n_out, self.n_out), "W_hr")
      self.b_r = init_bias(self.n_out, "b_r")

      self.W_xz = init_weight((self.n_in, self.n_out), "W_xz")
      self.W_hz = init_weight((self.n_out, self.n_out), "W_hz")
      self.b_z = init_bias(self.n_out, "b_z")

      self.W_xh = init_weight((self.n_in, self.n_out), "W_xh")
      self.W_hh = init_weight((self.n_out, self.n_out), "W_hh")
      self.b_h = init_bias(self.n_out, "b_h")

      self.params = [self.W_xr, self.W_hr, self.b_r,
                    self.W_xz, self.W_hz, self.b_z,
                    self.W_xh, self.W_hh, self.b_h]

      X = T.tensor3() # batch of sequence of vector
      Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null)
      def step_gru(x,pre_h):
         x = T.reshape(x, (batch_size, self.n_in))
         pre_h = T.reshape(pre_h, (batch_size, self.n_out))

         r = T.nnet.sigmoid(T.dot(x, self.W_xr) + T.dot(pre_h, self.W_hr) + self.b_r)
         z = T.nnet.sigmoid(T.dot(x, self.W_xz) + T.dot(pre_h, self.W_hz) + self.b_z)
         gh = T.tanh(T.dot(x, self.W_xh) + T.dot(r * pre_h, self.W_hh) + self.b_h)
         h = z * pre_h + (1 - z) * gh
         h = T.reshape(h, (1, batch_size * self.n_out))
         return h
      outputs, updates = theano.scan(step_gru, sequences = [X],
                                     outputs_info = [T.alloc(dtype(0.), 1, batch_size * self.n_out)])

      self.output = T.reshape(outputs, (X.shape[0], batch_size * self.n_out))

      cost = 0
      if cost_function == 'mse':
         cost = u.mse
      elif cost_function == 'cxe':
         cost = u.cxe
      else:
         cost = u.nll
      _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )

      self.train = theano.function(inputs=[X, Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
      self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
      self.n_param=(n_lstm*n_lstm*4+n_in*n_lstm*4+n_lstm*n_out+n_lstm*3)


