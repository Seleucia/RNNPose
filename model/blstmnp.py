import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_pweight,init_bias
from helper.optimizer import RMSprop
from theano import function as Tfunc

dtype = T.config.floatX

class blstmnp:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, output_activation=theano.tensor.nnet.relu,cost_function='nll',optimizer = RMSprop):
       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       #Forward weights
       self.W_xi_f = init_weight((self.n_in, self.n_lstm), 'W_xi', 'glorot')
       self.W_hi_f = init_weight((self.n_lstm, self.n_lstm), 'W_hi', 'ortho')
       self.b_i_f  = init_bias(self.n_lstm, sample='zero')
       self.W_xf_f = init_weight((self.n_in, self.n_lstm), 'W_xf', 'glorot')
       self.W_hf_f = init_weight((self.n_lstm, self.n_lstm), 'W_hf', 'ortho')
       self.b_f_f = init_bias(self.n_lstm, sample='one')
       self.W_xc_f = init_weight((self.n_in, self.n_lstm), 'W_xc', 'glorot')
       self.W_hc_f = init_weight((self.n_lstm, self.n_lstm), 'W_hc', 'ortho')
       self.b_c_f = init_bias(self.n_lstm, sample='zero')
       self.W_xo_f = init_weight((self.n_in, self.n_lstm), 'W_xo', 'glorot')
       self.W_ho_f = init_weight((self.n_lstm, self.n_lstm), 'W_ho', 'ortho')
       self.b_o_f = init_bias(self.n_lstm, sample='zero')
       self.W_hy_f = init_weight((self.n_lstm, self.n_out), 'W_hy', 'glorot')

       #Backward weights
       self.W_xi_b = init_weight((self.n_in, self.n_lstm), 'W_xi', 'glorot')
       self.W_hi_b = init_weight((self.n_lstm, self.n_lstm), 'W_hi', 'ortho')
       self.b_i_b  = init_bias(self.n_lstm, sample='zero')
       self.W_xf_b = init_weight((self.n_in, self.n_lstm), 'W_xf', 'glorot')
       self.W_hf_b = init_weight((self.n_lstm, self.n_lstm), 'W_hf', 'ortho')
       self.b_f_b = init_bias(self.n_lstm, sample='one')
       self.W_xc_b = init_weight((self.n_in, self.n_lstm), 'W_xc', 'glorot')
       self.W_hc_b = init_weight((self.n_lstm, self.n_lstm), 'W_hc', 'ortho')
       self.b_c_b = init_bias(self.n_lstm, sample='zero')
       self.W_xo_b = init_weight((self.n_in, self.n_lstm), 'W_xo', 'glorot')
       self.W_ho_b = init_weight((self.n_lstm, self.n_lstm), 'W_ho', 'ortho')
       self.b_o_b = init_bias(self.n_lstm, sample='zero')
       self.W_hy_b = init_weight((self.n_lstm, self.n_out), 'W_hy', 'glorot')

       self.b_y = init_bias(self.n_out, sample='zero')

       self.params = [self.W_xi_f, self.W_hi_f, self.b_i_f, self.W_xf_f,
                      self.W_hf_f, self.b_f_f,  self.W_xc_f, self.W_hc_f,
                      self.b_c_f, self.W_xo_f, self.W_ho_f, self.b_o_f,
                      self.W_hy_f,
                      self.W_xi_b, self.W_hi_b, self.b_i_b, self.W_xf_b,
                      self.W_hf_b, self.b_f_b,  self.W_xc_b, self.W_hc_b,
                      self.b_c_b, self.W_xo_b, self.W_ho_b, self.b_o_b,
                      self.W_hy_b, self.b_y]

       def f_step_lstm(x_t, h_tm1, c_tm1):
           i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi_f) + T.dot(h_tm1, self.W_hi_f) + self.b_i_f)
           f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf_f) + T.dot(h_tm1, self.W_hf_f) + self.b_f_f)
           c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc_f) + T.dot(h_tm1, self.W_hc_f) + self.b_c_f)
           o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo_f) + T.dot(h_tm1, self.W_ho_f) + self.b_o_f)
           h_t = o_t * T.tanh(c_t)
           return [h_t, c_t]

       def b_step_lstm(x_t, h_tm1, c_tm1):
           i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi_b) + T.dot(h_tm1, self.W_hi_b) + self.b_i_b)
           f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf_b) + T.dot(h_tm1, self.W_hf_b) + self.b_f_b)
           c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc_b) + T.dot(h_tm1, self.W_hc_b) + self.b_c_b)
           o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo_b) + T.dot(h_tm1, self.W_ho_b) + self.b_o_b)
           h_t = o_t * T.tanh(c_t)
           return [h_t, c_t]

       X_f = T.tensor3() # batch of sequence of vector
       X_b = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null)
       h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
       c0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

       [h_f, c_vals], _ = theano.scan(fn=f_step_lstm,
                                         sequences=X_f.dimshuffle(1,0,2),
                                         outputs_info=[h0, c0])

       [h_b, c_vals], _ = theano.scan(fn=b_step_lstm,
                                         sequences=X_b.dimshuffle(1,0,2),
                                         outputs_info=[h0, c0])

       h_b=h_b[:,::-1]
       y_vals=T.dot(h_f, self.W_hy_f)+T.dot(h_b, self.W_hy_b)+self.b_y

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


       self.train = theano.function(inputs=[X_f,X_b, Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X_f,X_b], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
       self.n_param=n_lstm*n_lstm*4+n_in*n_lstm*4+n_lstm*n_out+n_lstm*3
