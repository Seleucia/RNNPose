import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_pweight,init_bias,get_err_fn
from helper.optimizer import RMSprop

dtype = T.config.floatX

class bgru():
   def __init__(self, n_in, n_lstm, n_out, lr=0.00001, batch_size=64, output_activation=theano.tensor.nnet.relu,cost_function='mse',optimizer = RMSprop):
       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       self.W_xr_1 = init_weight((self.n_in, self.n_lstm), 'W_xr_1', 'glorot')
       self.W_hr_1 = init_weight((self.n_lstm, self.n_lstm), 'W_hr_1', 'ortho')
       self.b_r_1  = init_bias(self.n_lstm, sample='zero')
       self.W_xz_1 = init_weight((self.n_in, self.n_lstm), 'W_xz_1', 'glorot')
       self.W_hz_1 = init_weight((self.n_lstm, self.n_lstm), 'W_hz_1', 'ortho')
       self.b_z_1 = init_bias(self.n_lstm, sample='zero')
       self.W_xh_1 = init_weight((self.n_in, self.n_lstm), 'W_xh_1', 'glorot')
       self.W_hh_1 = init_weight((self.n_lstm, self.n_lstm), 'W_hh_1', 'ortho')
       self.b_h_1 = init_bias(self.n_lstm, sample='zero')
       # self.W_hy_1 = init_weight((self.n_lstm, self.n_out),'W_hy_1', 'glorot')
       # self.b_y_1 = init_bias(self.n_out, sample='zero')


       self.W_xr_2 = init_weight((self.n_in, self.n_lstm), 'W_xr', 'glorot')
       self.W_hr_2 = init_weight((self.n_lstm, self.n_lstm), 'W_hr', 'ortho')
       self.b_r_2  = init_bias(self.n_lstm, sample='zero')
       self.W_xz_2 = init_weight((self.n_in, self.n_lstm), 'W_xz', 'glorot')
       self.W_hz_2 = init_weight((self.n_lstm, self.n_lstm), 'W_hz', 'ortho')
       self.b_z_2 = init_bias(self.n_lstm, sample='zero')
       self.W_xh_2 = init_weight((self.n_in, self.n_lstm), 'W_xh', 'glorot')
       self.W_hh_2 = init_weight((self.n_lstm, self.n_lstm), 'W_hh', 'ortho')
       self.b_h_2 = init_bias(self.n_lstm, sample='zero')
       self.W_hy_2 = init_weight((self.n_lstm, self.n_out),'W_hy', 'glorot')
       self.b_y_2 = init_bias(self.n_out, sample='zero')

       self.params = [self.W_xr_1, self.W_hr_1, self.b_r_1,
                      self.W_xz_1, self.W_hz_1, self.b_z_1,
                      self.W_xh_1, self.W_hh_1, self.b_h_1,

                      self.W_xr_2, self.W_hr_2, self.b_r_2,
                      self.W_xz_2, self.W_hz_2, self.b_z_2,
                      self.W_xh_2, self.W_hh_2, self.b_h_2,
                      self.W_hy_f,self.W_hy_b, self.b_y
                      ]

       def f_step_lstm(x_t, h_tm1_1):
           r_t_1 = T.nnet.sigmoid(T.dot(x_t, self.W_xr_1) + T.dot(h_tm1_1, self.W_hr_1) + self.b_r_1)
           z_t_1 = T.nnet.sigmoid(T.dot(x_t, self.W_xz_1) + T.dot(h_tm1_1, self.W_hz_1)  + self.b_z_1)
           h_t_1 = T.tanh(T.dot(x_t, self.W_xh_1) + T.dot((r_t_1*h_tm1_1),self.W_hh_1)  + self.b_h_1)
           hh_t_1 = z_t_1 * h_t_1 + (1-z_t_1)*h_tm1_1
           return [hh_t_1]

       def b_step_lstm(x_t, h_tm1_2):
           r_t_2 = T.nnet.sigmoid(T.dot(x_t, self.W_xr_2) + T.dot(h_tm1_2, self.W_hr_2) + self.b_r_2)
           z_t_2 = T.nnet.sigmoid(T.dot(x_t, self.W_xz_2) + T.dot(h_tm1_2, self.W_hz_2)  + self.b_z_2)
           h_t_2 = T.tanh(T.dot(x_t, self.W_xh_2) + T.dot((r_t_2*h_tm1_2),self.W_hh_2)  + self.b_h_2)
           hh_t_2 = z_t_2 * h_t_2 + (1-z_t_2)*h_tm1_2
           return [hh_t_2]

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
       y_vals=T.tanh(T.dot(h_f, self.W_hy_f)+T.dot(h_b, self.W_hy_b)+self.b_y)


       self.output = y_vals.dimshuffle(1,0,2)

       cost=get_err_fn(self,cost_function,Y)

       _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )


       self.train = theano.function(inputs=[X_f,X_b, Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X_f,X_b], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
       self.n_param=n_lstm*n_lstm*3+n_in*n_lstm*3+n_lstm*n_out+n_lstm*3
