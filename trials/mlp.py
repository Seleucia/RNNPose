import time
import theano
import theano.tensor as T
import numpy as np
import numpy

def linear_rectified(x):
    y = T.maximum(T.cast(0., theano.config.floatX), x)
    return(y)

rng = np.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

def drop(input, p=0.5, rng=rng):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.

    """
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

def init_W_b(W, b, rng, n_in, n_out):

    # for a discussion of the initialization, see
    # https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa
    if W is None:
        W_values = numpy.asarray(
            rng.uniform(
                low=-np.sqrt(6./(n_in + n_out)),
                high=np.sqrt(6./(n_in + n_out)),
                size=(n_in, n_out)
                ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)

    # init biases to positive values, so we should be initially in the linear regime of the linear rectified function
    if b is None:
        b_values = numpy.ones((n_out,), dtype=theano.config.floatX) * np.cast[theano.config.floatX](0.01)
        b = theano.shared(value=b_values, name='b', borrow=True)
    return W, b

def rescale_weights(params, incoming_max):
    incoming_max = np.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * np.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)

class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None,
                 activation=linear_rectified, p=0.5):
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type is_train: theano.iscalar
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type p: float or double
        :param p: probability of NOT dropping out a unit
        """
        self.input = input
        # end-snippet-1

        W, b = init_W_b(W, b, rng, n_in, n_out)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        output = activation(lin_output)

        # multiply output and drop -> in an approximation the scaling effects cancel out
        train_output = drop(np.cast[theano.config.floatX](1./p) * output)

        #is_train is a pseudo boolean theano variable for switching between training and prediction
        self.output = T.switch(T.neq(is_train, 0), train_output, output)

        # parameters of the model
        self.params = [self.W, self.b]

from logistic_sgd import LogisticRegression, load_data

class DMLP(object):
    """Dropout resp. Dropconnect Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, is_train, input, n_in, n_hidden, n_out, DropXHiddenLayer = DropoutHiddenLayer, p=0.5):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type is_train: theano.iscalar
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type p: float or double
        :param p: probability of NOT dropping out a unit or connection

        """

        self.first_hiddenLayer = DropXHiddenLayer(
            rng=rng,
            is_train=is_train,
            input=input,
            n_in=n_in,
            n_out=n_hidden
        )

        # The second hidden layer gets as input the hidden units
        # of the first dropout hidden layer
        self.second_hiddenLayer = DropXHiddenLayer(
            rng=rng,
            is_train=is_train,
            input=self.first_hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_hidden
        )


        # The logistic regression layer gets as input the hidden units
        # of the second dropout hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.second_hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            #abs(self.first_hiddenLayer.W).sum() + abs(self.second_hiddenLayer.W).sum() +
            abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            #(self.first_hiddenLayer.W ** 2).sum() + (self.second_hiddenLayer.W ** 2).sum() +
            (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the layers it is
        # made out of
        self.params = self.first_hiddenLayer.params + self.second_hiddenLayer.params + self.logRegressionLayer.params

        # this is ugly from a software engineering point of view (stong coupling)
        self.params_rescale = [ self.first_hiddenLayer.params[0], self.second_hiddenLayer.params[0] ]


        self.consider_constant = []
        if hasattr(self.first_hiddenLayer, 'consider_constant'):
            self.consider_constant.extend(self.first_hiddenLayer.consider_constant)
        if hasattr(self.second_hiddenLayer, 'consider_constant'):
            self.consider_constant.extend(self.second_hiddenLayer.consider_constant)


def test_mlp(initial_learning_rate, initial_momentum = 0.5, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='/home/chris/data/mnist/mnist.pkl.gz', batch_size=20, n_hidden=500,
             DropXHiddenLayer = DropoutHiddenLayer, p=0.5):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight of the last layer when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight of the last layer when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file

    :type p: float or double
    :param p: probability of NOT dropping out a unit or connection

   """
    batch_size = np.cast['int32'](batch_size)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.iscalar()  # index to a [mini]batch
    #TODO compare with theano tutorial

    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction


    rng = numpy.random.RandomState(1234)

    # construct the Dropout MLP class
    classifier = DMLP(
        rng=rng,
        is_train=is_train,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        DropXHiddenLayer = DropXHiddenLayer
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + np.cast[theano.config.floatX](L1_reg) * classifier.L1
        + np.cast[theano.config.floatX](L2_reg) * classifier.L2_sqr
    )
    # end-snippet-4

    consider_constant = []

    if hasattr(classifier, 'consider_constant'):
        consider_constant.extend(classifier.consider_constant)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            is_train: np.cast['int32'](0)
        }
        #,updates = random_generator_updates
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            is_train: np.cast['int32'](0)
        }
        #,updates = random_generator_updates
    )



    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(np.cast[theano.config.floatX](initial_learning_rate) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert initial_momentum >= 0. and initial_momentum < 1.

    momentum =theano.shared(np.cast[theano.config.floatX](initial_momentum), name='momentum')

    updates = []
    for param in  classifier.params:
      param_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
      updates.append((param, param - learning_rate*param_update))
      updates.append((param_update, momentum*param_update + (np.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param, consider_constant=consider_constant)))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[T.cast(index, dtype='int32')],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            y: train_set_y[index * batch_size: (index + np.cast['int32'](1)) * batch_size],
            is_train: np.cast['int32'](1)
        }
    )

    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print "momentum: ", momentum.get_value()
        print "learning rate: ", learning_rate.get_value()
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            rescale_weights(classifier.params_rescale, 15.)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
        # adaption of momentum
        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
            momentum.set_value(np.cast[theano.config.floatX](new_momentum))
        # adaption of learning rate
        new_learning_rate = learning_rate.get_value() * 0.985
        learning_rate.set_value(np.cast[theano.config.floatX](new_learning_rate))


    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))


data_dir="/home/coskun/PycharmProjects/RNNPoseV2/data/mnist.pkl.gz"
# comment this out for a dropout experiment
test_mlp(initial_learning_rate=1.0, initial_momentum = 0.5, L1_reg=0.00, L2_reg=0.001,
         n_epochs=500, dataset=data_dir,
         batch_size=20, n_hidden=1000)
