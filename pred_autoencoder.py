import numpy as np
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import plot.plot_utils as pu
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
params= config.get_params()
params["model"]="autoencoder"
params['mfile']= "autoencoder_autoencoder_3_0.017986_best.p"
params["data_dir"]="/mnt/Data1/hc/img/"
# params["data_dir"]="/home/coskun/PycharmProjects/data/rnn/180k/"
#0, error 0.087360, 0.177598, 20.323595
#error 0.078438, 0.161955, 16.453038
#VAL--> epoch 21 | error 0.086701, 0.179906
rng = RandomStreams(seed=1234)
only_test=1
is_train=0
# params['seq_length']= 20
params['batch_size']=100
batch_size=params['batch_size']

sindex=0
(X_test,Y_test,F_list_test,G_list_test,S_Test_list)=du.load_pose(params,only_test=1,sindex=0)

n_test = len(X_test)
residual=n_test%batch_size
#residual=0
if residual>0:
   residual=batch_size-residual
   X_List=X_test.tolist()
   Y_List=Y_test.tolist()
   x=X_List[-1]
   y=Y_List[-1]
   f=F_list_test[-1]
   for i in range(residual):
      X_List.append(x)
      Y_List.append(y)
      F_list_test.append(f)
   X_test=np.asarray(X_List)
   Y_test=np.asarray(Y_List)
   n_test = len(Y_test)

# n_test_batches =n_test/ batch_size
n_test_batches = len(X_test)
n_test_batches /= batch_size

print("Test sample size: %i, Batch size: %i, test batch size: %i"%(X_test.shape[0]*X_test.shape[1],batch_size,n_test_batches))
print ("Model loading started")
model= model_provider.get_model_pretrained(params,rng)
print("Prediction started")
for minibatch_index in range(n_test_batches):
   x=X_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   y=Y_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   fl=F_list_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   pred = model.mid_layer(x,is_train)
   if residual>0:
      if(minibatch_index==n_test_batches-1):
         pred = pred[0:(len(pred)-residual)]
         y=y[0:(len(y)-residual)]
         fl=fl[0:(len(fl)-residual)]
   print(len(fl))
   u.write_auto_pred(pred,fl,params)
print "============================================================================"
# print fl
