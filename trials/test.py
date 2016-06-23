import numpy as np
import theano

mu, sigma = 0, 0.03 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
rng = RandomStreams(seed=1234)
import theano.tensor as T

noise= rng.normal(size=(1,1), std=0.0003, avg=0.0,dtype=theano.config.floatX)
# noise2=np.random.normal(loc=0.0, scale=0.001, size=(1,1))
x, y = T.dscalars('x', 'y')
z = x + y+noise

f = theano.function([x, y], z)

print(f(3,4))
print(f(3,4))
print(f(3,4))

print 'done'