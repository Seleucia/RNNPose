import numpy as np
import argparse
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import theano


def train_rnn(params):
   (X_train,Y_train,X_test,Y_test)=du.load_pose(params)
   params["len_train"]=len(X_train)
   params["len_test"]=len(X_test)
   u.start_log(params)
   batch_size=params['batch_size']
   n_train_batches = len(X_train)
   n_train_batches /= batch_size

   n_test_batches = len(X_test)
   n_test_batches /= batch_size

   nb_epochs=params['n_epochs']

   print("Batch size: %i, train batch size: %i, test batch size: %i"%(batch_size,n_train_batches,n_test_batches))
   u.log_write("Model build started",params)
   if params['resume']==1:
      model= model_provider.get_model_pretrained(params)
   else:
     model= model_provider.get_model(params)
   u.log_write("Number of parameters: %s"%(model.n_param),params)
   train_errors = np.ndarray(nb_epochs)
   u.log_write("Training started",params)
   val_counter=0
   for epoch_counter in range(nb_epochs):
      batch_loss = 0.
      for minibatch_index in range(n_train_batches):
          x=X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
          y=Y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]#60*20*54
          if(params["model"]=="blstmnp"):
             x_b=np.asarray(map(np.flipud,x))
             loss = model.train(x,x_b,y)
          else:
             loss = model.train(x, y)
          batch_loss += loss
      if params['shufle_data']==1:
         X_train,Y_train=du.shuffle_in_unison_inplace(X_train,Y_train)
      train_errors[epoch_counter] = batch_loss
      batch_loss/=n_train_batches
      s='TRAIN--> epoch %i | error %f'%(epoch_counter, batch_loss)
      u.log_write(s,params)
      best_loss=0
      if(epoch_counter%10==0):
          print("Model testing")
          batch_loss = 0.
          batch_loss3d = 0.
          for minibatch_index in range(n_test_batches):
             x=X_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
             y=Y_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
             if(params["model"]=="blstmnp"):
                x_b=np.asarray(map(np.flipud,x))
                pred = model.predictions(x,x_b)
             else:
                pred = model.predictions(x)

             loss=np.mean(np.abs(pred - y))
             loss3d =u.get_loss(y,pred)
             batch_loss += loss
             batch_loss3d += loss3d
          batch_loss/=n_test_batches
          batch_loss3d/=n_test_batches
          if(best_loss<batch_loss):
             best_loss=batch_loss
             ext=str(val_counter%5)+".p"
             u.write_params(model.params,params,ext)

          val_counter+=1
          s ='VAL--> epoch %i | error %f, %f '%(val_counter,batch_loss,batch_loss3d)
          u.log_write(s,params)


params= config.get_params()
parser = argparse.ArgumentParser(description='Training the module')
parser.add_argument('-m','--model',help='Model: lstm, lstm2, erd current('+params["model"]+')',default=params["model"])
args = vars(parser.parse_args())
params["model"]=args["model"]
params=config.update_params(params)
train_rnn(params)