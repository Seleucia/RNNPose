import numpy as np
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import plot.plot_utils as pu
params= config.get_params()
params["model"]="lstm"
params['mfile']= "lstm_model_test_1.p"
#0, error 0.087360, 0.177598, 20.323595
#error 0.078438, 0.161955, 16.453038
#VAL--> epoch 21 | error 0.086701, 0.179906



only_test=1
(X_test,Y_test,N_list)=du.load_pose(params,only_test)
BB_list=du.load_test_bboxes(params,len(N_list))
batch_size=params['batch_size']

n_test_batches = len(X_test)
n_test_batches /= batch_size
print("Batch size: %i, test batch size: %i"%(batch_size,n_test_batches))
print ("Model loading started")
model= model_provider.get_model_pretrained(params)
print("Prediction started")
batch_loss = 0.
batch_loss3d = 0.
batch_bb_loss = 0.
loss_list=[0]
for minibatch_index in range(n_test_batches):
   x=X_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   y=Y_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   n_list=N_list[minibatch_index * batch_size*y.shape[1]: (minibatch_index + 1) * batch_size*y.shape[1]]
   bb_list=BB_list[minibatch_index * batch_size*y.shape[1]: (minibatch_index + 1) * batch_size*y.shape[1]]
   pred = model.predictions(x)
   du.write_predictions(params,pred,n_list)
   loss=np.mean(np.abs(pred - y))
   (loss3d,bb_loss,l_list) =u.get_loss_bb(y,pred,bb_list)
   loss_list=loss_list+l_list
   batch_loss += loss
   batch_loss3d += loss3d
   batch_bb_loss+=bb_loss
batch_loss/=n_test_batches
batch_loss3d/=n_test_batches
batch_bb_loss/=n_test_batches
s ='error %f, %f, %f '%(batch_loss,batch_loss3d,batch_bb_loss)
print (s)
pu.plot_histograms(loss_list)
