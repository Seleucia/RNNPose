import numpy as np
import theano
import theano.tensor as T
from theano import shared
from helper.utils import init_weight
import helper.dt_utils as du

dtype = T.config.floatX

class RMSprop:
    def __init__(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):

        self.cost = cost
        self.params = params
        self.lr = shared(np.cast[dtype](lr))
        self.rho = shared(np.cast[dtype](rho))
        self.epsilon = shared(np.cast[dtype](epsilon))
        self.gparams = T.grad(self.cost, self.params)

    def getUpdates(self):

        acc = [shared(np.zeros(p.get_value(borrow=True).shape, dtype=dtype)) for p in self.params]
        updates = []

        for p, g, a in zip(self.params, self.gparams, acc):
            new_a = self.rho * a + (1 - self.rho) * (g ** 2)
            updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))

        return updates

class lstm:
   def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, single_output=True, output_activation=theano.tensor.nnet.relu,cost_function='nll'):
       self.n_in = n_in
       self.n_lstm = n_lstm
       self.n_out = n_out
       self.W_xi = init_weight((self.n_in, self.n_lstm),'W_xi')
       self.W_hi = init_weight((self.n_lstm, self.n_lstm),'W_hi', 'svd')
       self.W_ci = init_weight((self.n_lstm, self.n_lstm),'W_ci', 'svd')
       self.b_i = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
       self.W_xf = init_weight((self.n_in, self.n_lstm),'W_xf')
       self.W_hf = init_weight((self.n_lstm, self.n_lstm),'W_hf', 'svd')
       self.W_cf = init_weight((self.n_lstm, self.n_lstm),'W_cf', 'svd')
       self.b_f = shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_lstm)))
       self.W_xc = init_weight((self.n_in, self.n_lstm),'W_xc')
       self.W_hc = init_weight((self.n_lstm, self.n_lstm),'W_hc', 'svd')
       self.b_c = shared(np.zeros(n_lstm, dtype=dtype))
       self.W_xo = init_weight((self.n_in, self.n_lstm),'W_xo')
       self.W_ho = init_weight((self.n_lstm, self.n_lstm),'W_ho', 'svd')
       self.W_co = init_weight((self.n_lstm, self.n_lstm),'W_co', 'svd')
       self.b_o = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
       self.W_hy = init_weight((self.n_lstm, self.n_out),'W_hy')
       self.b_y = shared(np.zeros(n_out, dtype=dtype))
       self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                      self.W_xf, self.W_hf, self.W_cf, self.b_f,
                      self.W_xc, self.W_hc, self.b_c,
                      self.W_ho, self.W_co, self.b_o,
                      self.W_hy, self.b_y]


       def step_lstm(x_t, h_tm1, c_tm1):
           i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
           f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
           c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c)
           o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co)  + self.b_o)
           h_t = o_t * T.tanh(c_t)
           y_t = output_activation(T.dot(h_t, self.W_hy) + self.b_y)
           return [h_t, c_t, y_t]

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null)
       h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state
       c0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state

       [h_vals, c_vals, y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=X.dimshuffle(1,0,2),
                                         outputs_info=[h0, c0, None])

       if single_output:
           self.output = y_vals[-1]
       else:
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
       optimizer = RMSprop(
            cost,
            self.params,
            lr=lr
        )
       # gparams = T.grad(cost, self.params)
       # updates = OrderedDict()
       # for param, gparam in zip(self.params, gparams):
       #     updates[param] = param - gparam * lr
       # self.loss = theano.function(inputs = [X, Y], outputs = [cxe, mse, cost])
       # self.train = theano.function(inputs = [X, Y], outputs = cost, updates=updates,allow_input_downcast=True)

       self.train = theano.function(inputs=[X, Y],outputs=cost,updates=optimizer.getUpdates(),allow_input_downcast=True)

       #self.train = theano.function(inputs = [X, Y], outputs = cost, updates=updates,allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2),allow_input_downcast=True)
       self.debug = theano.function(inputs = [X, Y], outputs = [X.shape, Y.shape, y_vals.shape, cxe.shape])

(X_train,Y_train,X_test,Y_test)=du.laod_pose()
batch_size=64
n_train_batches = len(X_train)
n_train_batches /= batch_size

n_test_batches = len(X_test)
n_test_batches /= batch_size

print "Number of batches: "+str(n_train_batches)
print "Training size: "+str(len(X_train))

model = lstm(1024, 2, 54,batch_size=batch_size, single_output=False,output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
print("Model loaded")

lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
nb_epochs=250
train_errors = np.ndarray(nb_epochs)
def train_rnn():
 for i in range(nb_epochs):
   batch_loss = 0.
   for minibatch_index in range(n_train_batches):
       x=X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
       y=Y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
       loss = model.train(x, y)
       batch_loss += loss
   train_errors[i] = batch_loss
   batch_loss/=n_train_batches
   if(i%5==0):
       print("Model testing")
       test_rnn()
   print("Train Epoch loss(%s): %s"%(i,batch_loss))

def test_rnn():
    batch_loss = 0.
    for minibatch_index in range(n_test_batches):
       x=X_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
       y=Y_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
       pred = model.predictions(x)
       loss=np.mean((pred - y) ** 2)
       batch_loss += loss
    batch_loss/=n_test_batches
    print("Test Epoch loss: %s"%(batch_loss))

train_rnn()