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

class ConvLayer(object):
    def __init__(self, rng, input,filter_shape,input_shape,border_mode,subsample, activation=nn.relu):
        # e.g. input_shape= (samples, channels, rows, cols)
        #    assert border_mode in {'same', 'valid'}

        self.input = input
        nb_filter=filter_shape[0]

        W,b=None,None
        W =u.init_weight(filter_shape,rng=rng, name="w_conv", sample='glorot')
        b=u.init_bias(nb_filter,rng=rng)
        self.W = W
        self.b = b

        #image_shape: (batch size, num input feature maps,image height, image width)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=input_shape,
            border_mode=border_mode,subsample=subsample

        )
        output = conv_out + b.dimshuffle('x', 0, 'x', 'x')



        self.output = activation(output, 0)
        # parameters of the model
        self.params = [self.W, self.b]

        rows = input_shape[2]
        cols = input_shape[3]

        rows = u.conv_output_length(rows, filter_shape[2],border_mode, subsample[0])
        cols = u.conv_output_length(cols, filter_shape[3], border_mode, subsample[1])

        self.output_shape=(input_shape[0], nb_filter, rows, cols)

class DropoutLayer(object):
    def __init__(self, rng, input, prob):
        retain_prob = 1. - prob
        input *= rng.binomial(size=input.shape, p=retain_prob, dtype=input.dtype)
        # input /= retain_prob #Why dividing ?
        self.output= input

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

