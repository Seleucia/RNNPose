import theano
import theano.tensor as T
import numpy as np


shape=(1024,128)
fan_in, fan_out = shape[0],shape[1]
s = np.sqrt(2. / (fan_in + fan_out))
values=np.random.normal(loc=0.0, scale=s, size=shape)

# define tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

# define shared random stream
trng = T.shared_randomstreams.RandomStreams(1234)
d=trng.binomial(size=W[1].shape)

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym) * d, sequences=X)

results=results[::-1]

compute_with_bnoise = theano.function(inputs=[X, W, b_sym], outputs=[results],
                          updates=updates, allow_input_downcast=True)
x = np.eye(10, 2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
vals=compute_with_bnoise(x, w, b)
print(vals)