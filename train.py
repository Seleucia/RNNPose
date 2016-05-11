import numpy as np
from theano import shared
import theano.tensor as T
dtype = T.config.floatX
import argparse
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def train_rnn(params):
   rng = RandomStreams(seed=1234)
   (X_train,Y_train,S_Train_list,X_test,Y_test,S_Test_list)=du.load_pose(params)
   params["len_train"]=X_train.shape[0]*X_train.shape[1]
   params["len_test"]=X_test.shape[0]*X_test.shape[1]
   u.start_log(params)
   index_train_list,S_Train_list=du.get_batch_indexes(params,S_Train_list)
   index_test_list,S_Test_list=du.get_batch_indexes(params,S_Test_list)
   batch_size=params['batch_size']

   n_train_batches = len(index_train_list)
   n_train_batches /= batch_size

   n_test_batches = len(index_test_list)
   n_test_batches /= batch_size

   nb_epochs=params['n_epochs']

   print("Batch size: %i, train batch size: %i, test batch size: %i"%(batch_size,n_train_batches,n_test_batches))
   u.log_write("Model build started",params)
   if params['run_mode']==1:
      model= model_provider.get_model_pretrained(params,rng)
      u.log_write("Pretrained loaded: %s"%(params['mfile']),params)
   else:
     model= model_provider.get_model(params,rng)
   u.log_write("Number of parameters: %s"%(model.n_param),params)
   train_errors = np.ndarray(nb_epochs)
   u.log_write("Training started",params)
   val_counter=0
   best_loss=0.123376
   for epoch_counter in range(nb_epochs):
      batch_loss = 0.
      H=C=np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) # initial hidden state
      sid=0
      for minibatch_index in range(n_train_batches):
          id_lst=index_train_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
          tmp_sid=S_Train_list[(minibatch_index + 1) * batch_size-1]
          if(sid==0):
              sid=tmp_sid
          if(tmp_sid!=sid):
              sid=tmp_sid
              H=C=np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) # resetting initial state, since seq change
          x=X_train[id_lst] #60*20*1024
          y=Y_train[id_lst]#60*20*54
          is_train=1
          if(params["model"]=="blstmnp"):
             x_b=np.asarray(map(np.flipud,x))
             loss = model.train(x,x_b,y)
          else:
             loss,H,C= model.train(x, y,is_train,H,C)
          batch_loss += loss
      if params['shufle_data']==1:
         X_train,Y_train=du.shuffle_in_unison_inplace(X_train,Y_train)
      train_errors[epoch_counter] = batch_loss
      batch_loss/=n_train_batches
      s='TRAIN--> epoch %i | error %f'%(epoch_counter, batch_loss)
      u.log_write(s,params)
      if(epoch_counter%10==0):
          print("Model testing")
          batch_loss3d = []
          H=C=np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) # resetting initial state, since seq change
          sid=0
          for minibatch_index in range(n_test_batches):
             id_lst=index_test_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
             tmp_sid=S_Test_list[(minibatch_index + 1) * batch_size-1]
             if(sid==0):
                  sid=tmp_sid
             if(tmp_sid!=sid):
                  sid=tmp_sid
                  H=C=np.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) # resetting initial state, since seq change
             x=X_test[id_lst] #60*20*1024
             y=Y_test[id_lst]#60*20*54
             is_train=0
             if(params["model"]=="blstmnp"):
                x_b=np.asarray(map(np.flipud,x))
                pred = model.predictions(x,x_b)
             else:
                pred,H,C = model.predictions(x,is_train,H,C)
             loss3d =u.get_loss(y,pred)
             batch_loss3d.append(loss3d)
          batch_loss3d=np.nanmean(batch_loss3d)
          if(batch_loss3d<best_loss):
             best_loss=batch_loss3d
             ext=str(epoch_counter)+"_"+str(batch_loss3d)+"_best.p"
             u.write_params(model.params,params,ext)
          else:
              ext=str(val_counter%2)+".p"
              u.write_params(model.params,params,ext)

          val_counter+=1#0.08
          s ='VAL--> epoch %i | error %f, %f'%(val_counter,batch_loss3d,n_test_batches)
          u.log_write(s,params)


params= config.get_params()
parser = argparse.ArgumentParser(description='Training the module')
parser.add_argument('-m','--model',help='Model: lstm, lstm2, erd current('+params["model"]+')',default=params["model"])
args = vars(parser.parse_args())
params["model"]=args["model"]
params=config.update_params(params)
train_rnn(params)