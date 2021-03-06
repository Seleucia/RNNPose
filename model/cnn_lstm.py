from theano import shared
import numpy as np
import theano
import theano.tensor as T
from layers import ConvLayer,PoolLayer,HiddenLayer,DropoutLayer,LSTMLayer
import theano.tensor.nnet as nn
from helper.utils import init_weight,init_bias,get_err_fn,count_params, do_nothing
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
        is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

        #CNN global parameters.
        subsample=(1,1)
        p_1=0.5
        border_mode="valid"
        cnn_batch_size=batch_size*sequence_length
        pool_size=(2,2)

        #Layer1: conv2+pool+drop
        filter_shape=(64,1,9,9)
        input_shape=(cnn_batch_size,1,120,60) #input_shape= (samples, channels, rows, cols)
        input= X.reshape(input_shape)
        c1=ConvLayer(rng, input,filter_shape, input_shape,border_mode,subsample, activation=nn.relu)
        p1=PoolLayer(c1.output,pool_size=pool_size,input_shape=c1.output_shape)
        dl1=DropoutLayer(rng,input=p1.output,prob=p_1,is_train=is_train)
        retain_prob = 1. - p_1
        test_output = p1.output*retain_prob
        d1_output = T.switch(T.neq(is_train, 0), dl1.output, test_output)

        #Layer2: conv2+pool
        filter_shape=(128,p1.output_shape[1],3,3)
        c2=ConvLayer(rng, d1_output, filter_shape,p1.output_shape,border_mode,subsample, activation=nn.relu)
        p2=PoolLayer(c2.output,pool_size=pool_size,input_shape=c2.output_shape)


        #Layer3: conv2+pool
        filter_shape=(128,p2.output_shape[1],3,3)
        c3=ConvLayer(rng, p2.output,filter_shape,p2.output_shape,border_mode,subsample, activation=nn.relu)
        p3=PoolLayer(c3.output,pool_size=pool_size,input_shape=c3.output_shape)

        #Layer4: hidden
        n_in= reduce(lambda x, y: x*y, p3.output_shape[1:])
        x_flat = p3.output.flatten(2)
        h1=HiddenLayer(rng,x_flat,n_in,1024,activation=nn.relu)
        n_in=1024
        rnn_input = h1.output.reshape((batch_size,sequence_length, n_in))


        #Layer5: gru
        self.n_in = n_in
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_hy = init_weight((self.n_lstm, self.n_out), rng=rng,name='W_hy', sample= 'glorot')
        self.b_y = init_bias(self.n_out,rng=rng, sample='zero')

        layer1=LSTMLayer(rng,0,self.n_in,self.n_lstm)
        layer2=LSTMLayer(rng,1,self.n_lstm,self.n_lstm)
        layer3=LSTMLayer(rng,2,self.n_lstm,self.n_lstm)

        self.params = layer1.params+layer2.params+layer3.params
        self.params.append(self.W_hy)
        self.params.append(self.b_y)

        def step_lstm(x_t,mask,h_tm1_1,c_tm1_1,h_tm1_2,c_tm1_2,h_tm1_3,c_tm1_3):
           [h_t_1,c_t_1,y_t_1]=layer1.run(x_t,h_tm1_1,c_tm1_1)
           dl1=DropoutLayer(rng,input=y_t_1,prob=0.5,is_train=is_train,mask=mask)
           [h_t_2,c_t_2,y_t_2]=layer2.run(dl1.output,h_tm1_2,c_tm1_2)
           [h_t_3,c_t_3,y_t_3]=layer3.run(y_t_2,h_tm1_3,c_tm1_3)
           y = T.dot(y_t_3, self.W_hy) + self.b_y
           return [h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3,y]

        h0_1 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
        c0_1 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial cell state
        h0_2 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
        c0_2 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial cell state
        h0_3 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
        c0_3 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial cell state

        mask_shape=(sequence_length,batch_size,self.n_lstm)
        p_1=0.5
        mask= rng.binomial(size=mask_shape, p=p_1, dtype=X.dtype)

        #(1, 0, 2) -> AxBxC to BxAxC
        #(batch_size,sequence_length, n_in) >> (sequence_length, batch_size ,n_in)
        #T.dot(x_t, self.W_xi)x_t=(sequence_length, batch_size ,n_in), W_xi=  [self.n_in, self.n_lstm]

        [h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3,y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=[rnn_input.dimshuffle(1,0,2),mask],
                                         outputs_info=[h0_1, c0_1,h0_2, c0_2, h0_3, c0_3, None])

        self.output = y_vals.dimshuffle(1,0,2)

        self.params =c1.params+c2.params+c3.params+h1.params+self.params

        cost=get_err_fn(self,cost_function,Y)
        _optimizer = optimizer(cost, self.params, lr=lr)
        self.train = theano.function(inputs=[X,Y,is_train],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
        self.predictions = theano.function(inputs = [X,is_train], outputs = self.output,allow_input_downcast=True)
        self.n_param=count_params(self.params)