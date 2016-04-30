import helper.utils as u
from theano import shared
import numpy as np
import theano
import theano.tensor as T
from layers import LogisticRegression,PoolLayer,HiddenLayer,DropoutLayer,LSTMLayer
import theano.tensor.nnet as nn
from helper.utils import init_weight,init_bias,get_err_fn,count_params, do_nothing
from helper.optimizer import RMSprop

theano.config.exception_verbosity="high"
theano.config.optimizer='fast_compile'
dtype = T.config.floatX


class real_rcnn(object):
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
        border_mode="same"
        cnn_batch_size=batch_size*sequence_length
        pool_size=(2,2)


        n_lstm_layer=2
        f_dict=dict()
        f_dict['filter_shape_'+str(0)]=(64,1,9,9)
        f_dict['s_filter_shape_'+str(0)]=(64,f_dict['filter_shape_'+str(0)][0],9,9)

        f_dict['filter_shape_'+str(1)]=(128,f_dict['filter_shape_'+str(0)][0],3,3)
        f_dict['s_filter_shape_'+str(1)]=(128,f_dict['filter_shape_'+str(1)][0],3,3)

        f_dict['filter_shape_'+str(2)]=(128,f_dict['filter_shape_'+str(1)][0],3,3)
        f_dict['s_filter_shape_'+str(2)]=(128,f_dict['filter_shape_'+str(2)][0],3,3)


        input_shape=(batch_size,sequence_length,1,120,60) #input_shape= (samples, channels, rows, cols)
        input= X.reshape(input_shape).dimshuffle(1,0,2,3,4)
        input_shape=(batch_size,1,120,60) #input_shape= (samples, channels, rows, cols)

        s_dict=dict()
        layer_list=['p','l','p','d','l','p','l','p']
        rows=input_shape[2]
        cols=input_shape[3]
        counter=0
        outputs_info=[]
        for layer in (layer_list):
            if(layer=="l"):
                s_index=str(counter)
                if(counter==0):
                    pre_nfilter=input_shape[1]
                else:
                    pre_nfilter=f_dict['filter_shape_'+str(counter-1)][0]
                i_shape=(batch_size,pre_nfilter,rows,cols) #input_shape= (samples, channels, rows, cols)
                s_shape=(batch_size,f_dict['s_filter_shape_'+s_index][0],rows,cols) #input_shape= (samples, channels, rows, cols)
                h = shared(np.zeros(shape=s_shape, dtype=dtype)) # initial hidden state
                c = shared(np.zeros(shape=s_shape, dtype=dtype)) # initial hidden state
                s_dict["i_shape_"+s_index]=i_shape
                s_dict["s_shape_"+s_index]=s_shape
                outputs_info.append(h)
                outputs_info.append(c)
                counter+=1
            if(layer=="p"):
                rows=rows/2
                cols=cols/2
        s_dict["final_shape"]=(batch_size,sequence_length,pre_nfilter,rows,cols)

        outputs_info.append(None)
        outputs_info=tuple(outputs_info)

        p_dict=dict()
        for index in range(counter):
            s_index=str(index)
            p_dict['W_xi_'+s_index] =u.init_weight(f_dict['filter_shape_'+s_index],rng=rng, name="W_xi_"+s_index, sample='glorot')
            p_dict['W_hi_'+s_index] = u.init_weight(f_dict['s_filter_shape_'+s_index],rng=rng, name="W_hi_"+s_index, sample='glorot')
            p_dict['W_ci_'+s_index] = u.init_weight(f_dict['s_filter_shape_'+s_index],rng=rng, name="W_ci_"+s_index, sample='glorot')
            p_dict['b_i_'+s_index]=u.init_bias(f_dict['filter_shape_'+s_index][0],rng=rng,name="b_i_"+s_index)

            p_dict['W_xf_'+s_index] =u.init_weight(f_dict['filter_shape_'+s_index],rng=rng, name="W_xf_"+s_index, sample='glorot')
            p_dict['W_hf_'+s_index] = u.init_weight(f_dict['s_filter_shape_'+s_index],rng=rng, name="W_hf_"+s_index, sample='glorot')
            p_dict['W_cf_'+s_index] = u.init_weight(f_dict['s_filter_shape_'+s_index],rng=rng, name="W_cf_"+s_index, sample='glorot')
            p_dict['b_f_'+s_index]=u.init_bias(f_dict['filter_shape_'+s_index][0],rng=rng,name="b_f_"+s_index)

            p_dict['W_xc_'+s_index] =u.init_weight(f_dict['filter_shape_'+s_index],rng=rng, name="W_xc_"+s_index, sample='glorot')
            p_dict['W_hc_'+s_index] = u.init_weight(f_dict['s_filter_shape_'+s_index],rng=rng, name="W_hc_"+s_index, sample='glorot')
            p_dict['b_c_'+s_index]=u.init_bias(f_dict['filter_shape_'+s_index][0],rng=rng,name="b_c_"+s_index)

            p_dict['W_xo_'+s_index] =u.init_weight(f_dict['filter_shape_'+s_index],rng=rng, name="W_xo_"+s_index, sample='glorot')
            p_dict['W_ho_'+s_index] = u.init_weight(f_dict['s_filter_shape_'+s_index],rng=rng, name="W_ho_"+s_index, sample='glorot')
            p_dict['W_co_'+s_index] = u.init_weight(f_dict['s_filter_shape_'+s_index],rng=rng, name="W_co_"+s_index, sample='glorot')
            p_dict['b_o_'+s_index]=u.init_bias(f_dict['filter_shape_'+s_index][0],rng=rng,name="b_o_"+s_index)


        def step_lstm(x_t,mask,h_tm1_1,c_tm1_1,h_tm1_2,c_tm1_2,h_tm1_3,c_tm1_3):
           p1=PoolLayer(x_t,pool_size=pool_size,input_shape=s_dict["i_shape_0"])
           layer_1=LSTMLayer(rng,0,p_dict,f_dict,s_dict, p1.output,h_tm1_1,c_tm1_1,border_mode,subsample)
           [h_t_1,c_t_1,y_t_1]=layer_1.output
           p2=PoolLayer(y_t_1,pool_size=pool_size,input_shape=layer_1.yt_shape)
           dl1=DropoutLayer(rng,input=p2.output,prob=p_1,is_train=is_train,mask=mask)

           layer_2=LSTMLayer(rng,1,p_dict,f_dict,s_dict, dl1.output,h_tm1_2,c_tm1_2,border_mode,subsample)
           [h_t_2,c_t_2,y_t_2]=layer_2.output
           p2=PoolLayer(y_t_2,pool_size=pool_size,input_shape=layer_1.yt_shape)

           layer_3=LSTMLayer(rng,2,p_dict,f_dict,s_dict, p2.output,h_tm1_3,c_tm1_3,border_mode,subsample)
           [h_t_3,c_t_3,y_t_3]=layer_3.output
           p3=PoolLayer(y_t_3,pool_size=pool_size,input_shape=layer_3.yt_shape)


           return [h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3,p3.output]

        #(1, 0, 2) -> AxBxC to BxAxC
        #(batch_size,sequence_length, n_in) >> (sequence_length, batch_size ,n_in)
        #T.dot(x_t, self.W_xi_1)x_t=(sequence_length, batch_size ,n_in), W_xi_1=  [self.n_in, self.n_lstm]
        #5.293.568
        #185.983.658
        #19.100.202
        #8.090.154

        s_shape=list(s_dict["i_shape_1"])#after pooling filter sha[e
        s_shape.insert(0,sequence_length)
        mask_shape_1=tuple(s_shape)
        mask= rng.binomial(size=mask_shape_1, p=p_1, dtype=input.dtype)

        [h_t_1,c_t_1,h_t_2,c_t_2,h_t_3,c_t_3,y_vals],_ = theano.scan(fn=step_lstm,outputs_info=outputs_info,
                                         sequences=[input,mask])

        s_dict["final_shape"]=(batch_size,sequence_length,pre_nfilter,rows,cols)
        hidden_input=y_vals.dimshuffle(1,0,2,3,4)
        n_in= reduce(lambda x, y: x*y, s_dict["final_shape"][2:])
        x_flat = hidden_input.flatten(3)
        h1=HiddenLayer(rng,x_flat,n_in,1024,activation=nn.relu)
        n_in=1024
        lreg=LogisticRegression(rng,h1.output,n_in,42)

        self.output = lreg.y_pred
        self.params =p_dict.values()
        self.params.append(h1.params[0])
        self.params.append(h1.params[1])
        self.params.append(lreg.params[0])
        self.params.append(lreg.params[1])
        #
        # tmp = theano.tensor.switch(theano.tensor.isnan(Y),0,Y)
        cost=get_err_fn(self,cost_function,Y)
        L2_reg=0.0001
        L2_sqr = theano.shared(0.)
        for param in self.params:
            L2_sqr += (T.sum(param ** 2))

        cost += L2_reg*L2_sqr
        _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )
        # self.train = theano.function(inputs=[X,Y,is_train],outputs=cost,allow_input_downcast=True)
        #
        # _optimizer = optimizer(cost, self.params, lr=lr)
        self.train = theano.function(inputs=[X,Y,is_train],outputs=cost,updates=_optimizer.getUpdates(),allow_input_downcast=True)
        self.predictions = theano.function(inputs = [X,is_train], outputs = self.output,allow_input_downcast=True)
        self.n_param=count_params(self.params)