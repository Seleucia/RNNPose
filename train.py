import numpy as np
import theano
import helper.config as config
from helper.utils import init_weight
import model.model_provider as model_provider
import helper.dt_utils as du


def train_rnn(params):
   (X_train,Y_train,X_test,Y_test)=du.laod_pose()
   batch_size=params['batch_size']
   n_train_batches = len(X_train)
   n_train_batches /= batch_size

   n_test_batches = len(X_test)
   n_test_batches /= batch_size

   print "Number of batches: "+str(n_train_batches)
   print "Training size: "+str(len(X_train))
   nb_epochs=params['n_epochs']

   print("Model loaded")
   model=  model_provider.get_model(params)
   train_errors = np.ndarray(nb_epochs)
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
          batch_loss = 0.
          for minibatch_index in range(n_test_batches):
             x=X_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
             y=Y_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
             pred = model.predictions(x)
             loss=np.mean((pred - y) ** 2)
             batch_loss += loss
          batch_loss/=n_test_batches
          print("Test Epoch loss: %s"%(batch_loss))
          print("Train Epoch loss(%s): %s"%(i,batch_loss))


params= config.get_params()
train_rnn(params)