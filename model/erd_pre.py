import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_bias
from helper.optimizer import RMSprop

dtype = T.config.floatX


class erd_pre:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, single_output=True, output_activation=theano.tensor.tanh,cost_function='nll',optimizer = RMSprop):

       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       self.n_fc1=500
       self.n_fc2=500
       self.n_prefc1=256
       self.n_prefc2=256


       self.W_prefc1 = init_weight((self.n_in, self.n_prefc1),'W_prefc1')
       self.b_prefc1 = init_bias(self.n_prefc1, sample='zero')
       self.W_prefc2 = init_weight((self.n_prefc1, self.n_prefc2),'W_prefc2')
       self.b_prefc2 =init_bias(self.n_prefc2, sample='zero')

       self.W_fc1 = init_weight((self.n_fc1, self.n_fc2),'W_fc1')
       self.b_fc1 = init_bias(self.n_fc2, sample='zero')
       self.W_fc2 = init_weight((self.n_fc2, self.n_out),'W_fc2')
       self.b_fc2 =init_bias(self.n_out, sample='zero')

       self.W_xi = init_weight((self.n_prefc2, self.n_lstm),'W_xi')
       self.W_hi = init_weight((self.n_lstm, self.n_lstm),'W_hi', 'svd')
       self.W_ci = init_weight((self.n_lstm, self.n_lstm),'W_ci', 'svd')
       self.b_i = init_bias(self.n_lstm, sample='zero')
       self.W_xf = init_weight((self.n_prefc2, self.n_lstm),'W_xf')
       self.W_hf = init_weight((self.n_lstm, self.n_lstm),'W_hf', 'svd')
       self.W_cf = init_weight((self.n_lstm, self.n_lstm),'W_cf', 'svd')
       self.b_f =init_bias(self.n_lstm, sample='zero')
       self.W_xc = init_weight((self.n_prefc2, self.n_lstm),'W_xc')
       self.W_hc = init_weight((self.n_lstm, self.n_lstm),'W_hc', 'svd')
       self.b_c = shared(np.zeros(n_lstm, dtype=dtype))
       self.W_xo = init_weight((self.n_prefc2, self.n_lstm),'W_xo')
       self.W_ho = init_weight((self.n_lstm, self.n_lstm),'W_ho', 'svd')
       self.W_co = init_weight((self.n_lstm, self.n_lstm),'W_co', 'svd')
       self.b_o = init_bias(self.n_lstm, sample='zero')
       self.W_hy = init_weight((self.n_lstm, self.n_fc1),'W_hy')
       self.b_y = init_bias(self.n_fc1, sample='zero')

       self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                      self.W_xf, self.W_hf, self.W_cf, self.b_f,
                      self.W_xc, self.W_hc, self.b_c,
                      self.W_ho, self.W_co, self.b_o,
                      self.W_hy, self.b_y,self.W_fc1, self.b_fc1,self.W_fc2, self.b_fc2,
                      self.W_prefc1, self.b_prefc1,self.W_prefc2, self.b_prefc2]

       sigma = lambda x: 1 / (1 + T.exp(-x))

       def step_lstm(x_t, h_tm1, c_tm1):
           i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
           f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
           c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c)
           o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co)  + self.b_o)
           h_t = o_t * T.tanh(c_t)
           y_t = T.nnet.sigmoid(T.dot(h_t, self.W_hy) + self.b_y)
           return [h_t, c_t, y_t]

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null)
       h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
       c0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

       #Hidden layer
       prefc1_out = T.tanh(T.dot(X.dimshuffle(1,0,2), self.W_prefc1)  + self.b_prefc1)
       prefc2_out = T.tanh(T.dot(prefc1_out, self.W_prefc2)  + self.b_prefc2)

       [h_vals, c_vals, y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=prefc2_out,
                                         outputs_info=[h0, c0, None])


       #Hidden layer
       fc1_out = T.tanh(T.dot(y_vals, self.W_fc1)  + self.b_fc1)
       fc2_out = T.tanh(T.dot(fc1_out, self.W_fc2)  + self.b_fc2)

       self.output=fc2_out.dimshuffle(1,0,2)

       cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
       nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))
       mse = T.mean((self.output - Y) ** 2)

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
       self.predictions = theano.function(inputs = [X], outputs = self.output,allow_input_downcast=True)
       self.n_param=n_lstm*n_lstm*4+n_in*n_lstm*4+n_lstm*n_out+n_lstm*3
