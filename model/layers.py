import helper.utils as u
import theano.tensor.signal.pool as pool
import theano.tensor.nnet as nn
import theano.tensor as T
from theano.tensor.nnet import conv2d #Check if it is using gpu or not

class LogisticRegression(object):
    def __init__(self,  rng,input, n_in, n_out):
        shape=(n_in, n_out)
        self.W = u.init_weight(shape=shape,rng=rng,name='W_xif',sample='glorot')
        self.b=u.init_bias(n_out,rng=rng)

        self.y_pred = T.dot(input, self.W) + self.b
        self.params = [self.W, self.b]
        self.input = input


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,activation=T.tanh):
        self.input = input
        shape=[n_in,n_out]
        W =u.init_weight(shape=shape,rng=rng,name="w_hid",sample="glorot")
        b=u.init_bias(n_out,rng)
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]

class LSTMLayer(object):
    def __init__(self, rng,layer_id,p_dict,f_dict,s_dict,x_t,h_tm1,c_tm1,border_mode,subsample):
        layer_id=str(layer_id)

        x_i_t=ConvLayer(rng, x_t,f_dict['filter_shape_'+layer_id],s_dict["i_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_xi_'+layer_id],only_conv=1)
        h_i_t=ConvLayer(rng, h_tm1,f_dict['s_filter_shape_'+layer_id],s_dict["s_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_hi_'+layer_id],only_conv=1)
        c_i_t=ConvLayer(rng, c_tm1,f_dict['s_filter_shape_'+layer_id],s_dict["s_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_ci_'+layer_id],only_conv=1)
        i_t=T.nnet.sigmoid((x_i_t.output+h_i_t.output+c_i_t.output)+p_dict['b_i_'+layer_id].dimshuffle('x', 0, 'x', 'x'))

        x_f_t=ConvLayer(rng, x_t,f_dict['filter_shape_'+layer_id],s_dict["i_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_xf_'+layer_id],only_conv=1)
        h_f_t=ConvLayer(rng, h_tm1,f_dict['s_filter_shape_'+layer_id],s_dict["s_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_hf_'+layer_id],only_conv=1)
        c_f_t=ConvLayer(rng, c_tm1,f_dict['s_filter_shape_'+layer_id],s_dict["s_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_cf_'+layer_id],only_conv=1)
        f_t=T.nnet.sigmoid((x_f_t.output+h_f_t.output+c_f_t.output)+p_dict['b_f_'+layer_id].dimshuffle('x', 0, 'x', 'x'))

        x_c_t=ConvLayer(rng, x_t,f_dict['filter_shape_'+layer_id],s_dict["i_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_xc_'+layer_id],only_conv=1)
        h_c_t=ConvLayer(rng, h_tm1,f_dict['s_filter_shape_'+layer_id],s_dict["s_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_hc_'+layer_id],only_conv=1)
        c_t = f_t * c_tm1 + i_t * T.tanh(x_c_t.output + h_c_t.output + p_dict['b_c_'+layer_id].dimshuffle('x', 0, 'x', 'x'))

        x_o_t=ConvLayer(rng, x_t,f_dict['filter_shape_'+layer_id],s_dict["i_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_xo_'+layer_id],only_conv=1)
        h_o_t=ConvLayer(rng, h_tm1,f_dict['s_filter_shape_'+layer_id],s_dict["s_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_ho_'+layer_id],only_conv=1)
        c_o_t=ConvLayer(rng, c_t,f_dict['s_filter_shape_'+layer_id],s_dict["s_shape_"+layer_id], border_mode, subsample, activation=nn.relu,W=p_dict['W_co_'+layer_id],only_conv=1)
        o_t = T.nnet.sigmoid(x_o_t.output+ h_o_t.output + c_o_t.output  + p_dict['b_o_'+layer_id].dimshuffle('x', 0, 'x', 'x'))
        h_t = o_t * T.tanh(c_t)
        y_t =h_t
        self.yt_shape=x_i_t.output_shape
        self.output=[h_t,c_t,y_t]


class ConvLayer(object):
    def __init__(self, rng, input,filter_shape,input_shape,border_mode,subsample, activation=nn.relu,W=None,b=None,only_conv=0):
        # e.g. input_shape= (samples, channels, rows, cols)
        #    assert border_mode in {'same', 'valid'}

        self.input = input
        nb_filter=filter_shape[0]

        # W,b=None,None
        if(W ==None):
            W =u.init_weight(filter_shape,rng=rng, name="w_conv", sample='glorot')
            b=u.init_bias(nb_filter,rng=rng)
        self.W = W
        self.b = b

        b_mode=border_mode
        if(border_mode=='same'):
            b_mode='half'

        #image_shape: (batch size, num input feature maps,image height, image width)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=input_shape,
            border_mode=b_mode,subsample=subsample

        )

        if border_mode == 'same':
            if filter_shape[2] % 2 == 0:
                conv_out = conv_out[:, :, :(input.shape[2] + subsample[0] - 1) // subsample[0], :]
            if filter_shape[3] % 2 == 0:
                conv_out = conv_out[:, :, :, :(input.shape[3] + subsample[1] - 1) // subsample[1]]

        if(only_conv==0):
            output = conv_out + b.dimshuffle('x', 0, 'x', 'x')
            self.output = activation(output, 0)
        else:
            self.output = conv_out


        # parameters of the model
        self.params = [self.W, self.b]

        rows = input_shape[2]
        cols = input_shape[3]

        rows = u.conv_output_length(rows, filter_shape[2],border_mode, subsample[0])
        cols = u.conv_output_length(cols, filter_shape[3], border_mode, subsample[1])

        self.output_shape=(input_shape[0], nb_filter, rows, cols)

class DropoutLayer(object):
    def __init__(self, rng, input, prob,is_train,mask=None):
        retain_prob = 1. - prob
        if(mask==None):
            input *= rng.binomial(size=input.shape, p=retain_prob, dtype=input.dtype)
        else:
            input *= mask

        ret_input =input/ retain_prob #Why dividing ?
        test_output = input*retain_prob
        self.output= T.switch(T.neq(is_train, 0), ret_input, test_output)


class PoolLayer(object):
    def __init__(self, input,pool_size,input_shape, pool_mode="max"):
        pooled_out = pool.pool_2d(
            input=input,
            ds=pool_size,
            mode=pool_mode,
            ignore_border=True
        )


        rows = input_shape[2]
        cols = input_shape[3]
        rows = rows /  pool_size[0]
        cols = cols /  pool_size[1]
        self.output_shape= (input_shape[0], input_shape[1], rows, cols)

        self.output= pooled_out

