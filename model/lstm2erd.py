import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_pweight,init_bias,get_err_fn
from helper.optimizer import RMSprop

dtype = T.config.floatX

class lstm2erd:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, output_activation=theano.tensor.nnet.relu,cost_function='nll',optimizer = RMSprop):
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

       #1th layer
       self.W_xi_1 = init_weight((self.n_in, self.n_lstm), 'W_xi_1', 'glorot')
       self.W_hi_1 = init_weight((self.n_lstm, self.n_lstm), 'W_hi_1', 'glorot')
       self.W_ci_1 = init_weight((self.n_lstm, self.n_lstm), 'W_ci_1', 'glorot')
       self.b_i_1 = init_bias(self.n_lstm, sample='zero')
       self.W_xf_1 = init_weight((self.n_in, self.n_lstm), 'W_xf_1', 'glorot')
       self.W_hf_1 = init_weight((self.n_lstm, self.n_lstm), 'W_hf_1', 'glorot')
       self.W_cf_1 = init_weight((self.n_lstm, self.n_lstm), 'W_cf_1', 'glorot')
       self.b_f_1 = init_bias(self.n_lstm, sample='one')
       self.W_xc_1 = init_weight((self.n_in, self.n_lstm), 'W_xc_1', 'glorot')
       self.W_hc_1 = init_weight((self.n_lstm, self.n_lstm), 'W_hc_1', 'glorot')
       self.b_c_1 = init_bias(self.n_lstm, sample='zero')
       self.W_xo_1 = init_weight((self.n_in, self.n_lstm), 'W_xo_1', 'glorot')
       self.W_ho_1 = init_weight((self.n_lstm, self.n_lstm), 'W_ho_1', 'glorot')
       self.W_co_1 = init_weight((self.n_lstm, self.n_lstm), 'W_co_1', 'glorot')
       self.b_o_1 = init_bias(self.n_lstm, sample='zero')
       #self.W_hy_1 = init_weight((self.n_lstm, self.n_out), 'W_hy_1')
       #self.b_y_1 = init_bias(self.n_lstm, sample='zero')

       #2th layer
       self.W_xi_2 = init_weight((self.n_lstm, self.n_lstm), 'W_xi_2', 'glorot')
       self.W_hi_2 = init_weight((self.n_lstm, self.n_lstm), 'W_hi_2', 'glorot')
       self.W_ci_2 = init_weight((self.n_lstm, self.n_lstm), 'W_ci_2', 'glorot')
       self.b_i_2 = init_bias(self.n_lstm, sample='zero')
       self.W_xf_2 = init_weight((self.n_lstm, self.n_lstm), 'W_xf_2', 'glorot')
       self.W_hf_2 = init_weight((self.n_lstm, self.n_lstm), 'W_hf_2', 'glorot')
       self.W_cf_2 = init_weight((self.n_lstm, self.n_lstm), 'W_cf_2', 'glorot')
       self.b_f_2 = init_bias(self.n_lstm, sample='one')
       self.W_xc_2 = init_weight((self.n_lstm, self.n_lstm), 'W_xc_2', 'glorot')
       self.W_hc_2 = init_weight((self.n_lstm, self.n_lstm), 'W_hc_2', 'glorot')
       self.b_c_2 = init_bias(self.n_lstm, sample='zero')
       self.W_xo_2 = init_weight((self.n_lstm, self.n_lstm), 'W_xo_2', 'glorot')
       self.W_ho_2 = init_weight((self.n_lstm, self.n_lstm), 'W_ho_2', 'glorot')
       self.W_co_2 = init_weight((self.n_lstm, self.n_lstm), 'W_co_2', 'glorot')
       self.b_o_2 = init_bias(self.n_lstm, sample='zero')
       self.W_hy_2 = init_weight((self.n_lstm, self.n_out), 'W_hy_2', 'glorot')
       self.b_y_2 = init_bias(self.n_out, sample='zero')


       self.params = [
                      self.W_xi_1, self.W_hi_1, self.W_ci_1, self.b_i_1,
                      self.W_xf_1, self.W_hf_1, self.W_cf_1, self.b_f_1,
                      self.W_xc_1, self.W_hc_1, self.b_c_1, self.W_xo_1, self.W_ho_1,
                      self.W_co_1, self.b_o_1, # self.W_hy_1, self.b_y_1,
                      self.W_xi_2, self.W_hi_2, self.W_ci_2, self.b_i_2,
                      self.W_xf_2, self.W_hf_2, self.W_cf_2, self.b_f_2,
                      self.W_xc_2, self.W_hc_2, self.b_c_2, self.W_xo_2,  self.W_ho_2,
                      self.W_co_2, self.b_o_2,  self.W_hy_2, self.b_y_2,
                      self.W_fc1, self.b_fc1,self.W_fc2, self.b_fc2,self.W_fc3, self.b_fc3
                      ]


       def step_lstm(x_t, h_tm1, c_tm1,h_tm2,c_tm2):
           i_t_1 = T.nnet.sigmoid(T.dot(x_t, self.W_xi_1) + T.dot(h_tm1, self.W_hi_1) + T.dot(c_tm1, self.W_ci_1) + self.b_i_1)
           f_t_1 = T.nnet.sigmoid(T.dot(x_t, self.W_xf_1) + T.dot(h_tm1, self.W_hf_1) + T.dot(c_tm1, self.W_cf_1) + self.b_f_1)
           c_t_1 = f_t_1 * c_tm1 + i_t_1 * T.tanh(T.dot(x_t, self.W_xc_1) + T.dot(h_tm1, self.W_hc_1) + self.b_c_1)
           o_t_1 = T.nnet.sigmoid(T.dot(x_t, self.W_xo_1) + T.dot(h_tm1, self.W_ho_1) + T.dot(c_t_1, self.W_co_1) + self.b_o_1)
           h_t_1 = o_t_1 * T.tanh(c_t_1)
           #y_t_1 = output_activation(T.dot(h_t_1, self.W_hy_1) + self.b_y_1)

           i_t_2 = T.nnet.sigmoid(T.dot(h_t_1, self.W_xi_2) + T.dot(h_tm2, self.W_hi_2) + T.dot(c_tm2, self.W_ci_2) + self.b_i_2)
           f_t_2 = T.nnet.sigmoid(T.dot(h_t_1, self.W_xf_2) + T.dot(h_tm2, self.W_hf_2) + T.dot(c_tm2, self.W_cf_2) + self.b_f_2)
           c_t_2 = f_t_2 * c_tm2 + i_t_2 * T.tanh(T.dot(h_t_1, self.W_xc_2) + T.dot(h_tm2, self.W_hc_2) + self.b_c_2)
           o_t_2 = T.nnet.sigmoid(T.dot(h_t_1, self.W_xo_2) + T.dot(h_tm2, self.W_ho_2) + T.dot(c_t_2, self.W_co_2) + self.b_o_2)
           h_t_2 = o_t_2 * T.tanh(c_t_2)
           y_t_2 = T.tanh(T.dot(h_t_2, self.W_hy_2) + self.b_y_2)

           return [h_t_1,c_t_1,h_t_2,c_t_2, y_t_2]

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null)
       h0_1 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
       c0_1 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
       h0_2 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
       c0_2 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

       [h_vals_1, c_vals_1,h_vals_2, c_vals_2, y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=X.dimshuffle(1,0,2),
                                         outputs_info=[h0_1, c0_1,h0_2, c0_2, None])
       #Hidden layers
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
       self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
       self.n_param=(n_lstm*n_lstm*4+n_in*n_lstm*4+n_lstm*n_out+n_lstm*3)*2