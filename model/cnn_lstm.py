import glob
from theano import shared
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
import theano.tensor as T
from layers import ConvLayer,PoolLayer,LogisticRegression,DropoutLayer
import theano.tensor.nnet as nn
from helper.utils import init_weight,init_bias,get_err_fn
from helper.optimizer import RMSprop

# theano.config.exception_verbosity="high"
dtype = T.config.floatX

class cnn_lstm(object):
    def __init__(self,rng,params,cost_function='mse',optimizer = RMSprop):

        lr=params["lr"]
        n_lstm=params['n_hidden']
        n_out=params['n_output']
        batch_size=params["batch_size"]
        sequence_length=params["seq_length"]

        # minibatch)
        X = T.tensor3() # batch of sequence of vector
        Y = T.tensor3() # batch of sequence of vector



        cnn_batch_size=batch_size*sequence_length
        pool_size=(2,2)
        nb_filter=64
        nb_row=9
        nb_col=9
        filter_shape=(9,9)
        border_mode="valid"
        subsample=(1,1)
        input_shape=(cnn_batch_size,1,120,60) #input_shape= (samples, channels, rows, cols)
        input= X.reshape(input_shape)
        c1=ConvLayer(rng, input, nb_filter, nb_row, nb_col,input_shape,filter_shape,border_mode,subsample, activation=nn.relu)
        p1=PoolLayer(c1.output,pool_size=pool_size,input_shape=c1.output_shape,border_mode=border_mode)
        dl1=DropoutLayer(rng,input=p1.output,prob=0.5)
        #
        #
        nb_filter=128
        nb_row=3
        nb_col=3
        filter_shape=(3,3)
        subsample=(1,1)
        c2=ConvLayer(rng, dl1.output, nb_filter, nb_row, nb_col,p1.output_shape,filter_shape,border_mode,subsample, activation=nn.relu)
        p2=PoolLayer(c2.output,pool_size=pool_size,input_shape=c2.output_shape,border_mode=border_mode)
        #
        #
        nb_filter=128
        nb_row=3
        nb_col=3
        filter_shape=(3,3)
        subsample=(1,1)
        c3=ConvLayer(rng, p2.output, nb_filter, nb_row, nb_col,p2.output_shape,filter_shape,border_mode,subsample, activation=nn.relu)
        p3=PoolLayer(c3.output,pool_size=pool_size,input_shape=c3.output_shape,border_mode=border_mode)
        #
        #
        # nb_filter=512
        # nb_row=13
        # nb_col=5
        # filter_shape=(13,5)
        # subsample=(1,1)
        # c4=ConvLayer(rng, p3.output, nb_filter, nb_row, nb_col,p3.output_shape,filter_shape,border_mode,subsample, activation=nn.relu)
        # p4=PoolLayer(c4.output,pool_size=pool_size,input_shape=c4.output_shape,border_mode=border_mode)
        #

        n_in= reduce(lambda x, y: x*y, p3.output_shape[1:])
        x_flat = p3.output.flatten(2)

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

        rnn_input = x_flat.reshape((batch_size,sequence_length, n_in))
        h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

        #(1, 0, 2) -> AxBxC to BxAxC
        #(batch_size,sequence_length, n_in) >> (sequence_length, batch_size ,n_in)
        # ValueError: Input dimension mis-match. (input[0].shape[0] = 2(bs), input[1].shape[0] = 8(n_lstm))
        #T.dot(x_t, self.W_xi)x_t=(sequence_length, batch_size ,n_in), W_xi=  [self.n_in, self.n_lstm]

        [h_vals, y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=rnn_input.dimshuffle(1,0,2),
                                         outputs_info=[h0, None])

        self.output = y_vals.dimshuffle(1,0,2)

        self.params =self.params+c1.params+c2.params+c3.params

        cost=get_err_fn(self,cost_function,Y)
        _optimizer = optimizer(cost, self.params, lr=lr)
        self.train = theano.function(inputs=[X,Y],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
        self.predictions = theano.function(inputs = [X], outputs = self.output,allow_input_downcast=True)
        self.n_param=n_lstm*n_lstm*4+n_in*n_lstm*4+n_lstm*n_out+n_lstm*3