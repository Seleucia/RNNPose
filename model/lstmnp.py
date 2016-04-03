import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_bias
from helper.optimizer import RMSprop

dtype = T.config.floatX

class lstmnp:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, output_activation=theano.tensor.nnet.relu,cost_function='mse',optimizer = RMSprop):
       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       self.W_xi = init_weight((self.n_in, self.n_lstm),'W_xi', 'glorot')
       self.W_hi = init_weight((self.n_lstm, self.n_lstm),'W_hi', 'ortho')
       self.b_i  = init_bias(self.n_lstm, sample='zero')
       self.W_xf = init_weight((self.n_in, self.n_lstm),'W_xf', 'glorot')
       self.W_hf = init_weight((self.n_lstm, self.n_lstm),'W_hf', 'ortho')
       self.b_f = init_bias(self.n_lstm, sample='one')
       self.W_xc = init_weight((self.n_in, self.n_lstm),'W_xc', 'glorot')
       self.W_hc = init_weight((self.n_lstm, self.n_lstm),'W_hc', 'ortho')
       self.b_c = init_bias(self.n_lstm, sample='zero')
       self.W_xo = init_weight((self.n_in, self.n_lstm),'W_xo', 'glorot')
       self.W_ho = init_weight((self.n_lstm, self.n_lstm),'W_ho', 'ortho')
       self.b_o = init_bias(self.n_lstm, sample='zero')
       self.W_hy = init_weight((self.n_lstm, self.n_out),'W_hy', 'glorot')
       self.b_y = init_bias(self.n_out, sample='zero')

       self.params = [self.W_xi, self.W_hi, self.b_i,
                      self.W_xf, self.W_hf,  self.b_f,
                      self.W_xc, self.W_hc, self.b_c,  self.W_xo,
                      self.W_ho, self.b_o,
                      self.W_hy, self.b_y]

       def step_lstm(x_t, h_tm1, c_tm1):
           i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + self.b_i)
           f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + self.b_f)
           c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c)
           o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + self.b_o)
           h_t = o_t * T.tanh(c_t)
           y_t = T.tanh(T.dot(h_t, self.W_hy) + self.b_y)
           return [h_t, c_t, y_t]

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null)
       h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
       c0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

       [h_vals, c_vals, y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=X.dimshuffle(1,0,2),
                                         outputs_info=[h0, c0, None])

       self.output = y_vals.dimshuffle(1,0,2)
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
       def reset():
           h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype))
           c0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype))


       self.train = theano.function(inputs=[X, Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
       self.n_param=n_lstm*n_lstm*4+n_in*n_lstm*4+n_lstm*n_out+n_lstm*3
