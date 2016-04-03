import helper.utils as u
from theano.tensor.signal import pool
import theano.tensor.signal.downsample as downsample
import theano.tensor.nnet as nn
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv2d #Check if it is using gpu or not

class LogisticRegression(object):
    def __init__(self,  rng,input, n_in, n_out):
        shape=(n_in, n_out)
        self.W = u.init_weight(shape=shape,rng=rng,name='W_xif',sample='glorot')
        self.b=u.init_bias(n_out,rng=rng)

        self.y_pred = T.tanh(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]
        self.input = input


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,activation=nn.relu):
        self.input = input
        shape=[n_in,n_out]
        W =u.init_weight(shape=shape,rng=rng,name="w_hid",sample="glorot")
        b=u.init_bias(n_out,rng)
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output, 0)
        # parameters of the model
        self.params = [self.W, self.b]

class ConvLayer(object):
    def __init__(self, rng, input, nb_filter, nb_row, nb_col,input_shape,filter_shape,border_mode,subsample, activation=nn.relu):
        # e.g. input_shape= (samples, channels, rows, cols)
        #    assert border_mode in {'same', 'valid'}

        self.input = input
        shape= (nb_filter, input_shape[1], nb_row, nb_col)

        W,b=None,None
        W =u.init_weight(shape,rng=rng, name="w_conv", sample='glorot')
        b=u.init_bias(nb_filter,rng=rng)
        self.W = W
        self.b = b

        #image_shape: (batch size, num input feature maps,image height, image width)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=input_shape

        )
        output = conv_out + b.dimshuffle('x', 0, 'x', 'x')



        self.output = activation(output, 0)
        # parameters of the model
        self.params = [self.W, self.b]

        rows = input_shape[2]
        cols = input_shape[3]

        rows = u.conv_output_length(rows, nb_row,border_mode, subsample[0])
        cols = u.conv_output_length(cols, nb_col, border_mode, subsample[1])

        self.output_shape=(input_shape[0], nb_filter, rows, cols)

class DropoutLayer(object):
    def __init__(self, rng, input, prob):
        retain_prob = 1. - prob
        input *= rng.binomial(size=input.shape, p=retain_prob, dtype=input.dtype)
        input /= retain_prob #Why dividing ?
        self.output= input

class PoolLayer(object):
    def __init__(self, input,pool_size,input_shape,strides=(1, 1),border_mode = 'same', pool_mode="max"):
        pooled_out = downsample.max_pool_2d(
            input=input,
            ds=pool_size,
            ignore_border=True
        )


        rows = input_shape[2]
        cols = input_shape[3]
        rows = rows /  pool_size[0]
        cols = cols /  pool_size[1]

        self.output_shape= (input_shape[0], input_shape[1], rows, cols)

        self.output= pooled_out

