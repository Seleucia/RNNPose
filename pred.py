import numpy as np
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import plot.plot_utils as pu
params= config.get_params()
params["model"]="erd"
params['mfile']= "erd_model_test_4_xx.p"
#0, error 0.087360, 0.177598, 20.323595
#error 0.078438, 0.161955, 16.453038
#VAL--> epoch 21 | error 0.086701, 0.179906

only_test=1
params['seq_length']= 20
params['batch_size']=62
batch_size=params['batch_size']

u.prep_pred_file(params)

(X_test,Y_test,N_list,G_list)=du.load_pose(params,only_test)

n_test = len(X_test)
residual=n_test%batch_size
#residual=0
if residual>0:
   residual=batch_size-residual
   X_List=X_test.tolist()
   Y_List=Y_test.tolist()
   x=X_List[-1]
   y=Y_List[-1]
   for i in range(residual):
      X_List.append(x)
      Y_List.append(y)
   X_test=np.asarray(X_List)
   Y_test=np.asarray(Y_List)
   n_test = len(Y_test)

# n_test_batches =n_test/ batch_size
n_test_batches = len(X_test)
n_test_batches /= batch_size

print("Test sample size: %i, Batch size: %i, test batch size: %i"%(X_test.shape[0]*X_test.shape[1],batch_size,n_test_batches))
print ("Model loading started")
model= model_provider.get_model_pretrained(params)
print("Prediction started")
batch_loss = 0.
batch_loss3d = 0.
batch_bb_loss = 0.
loss_list=[0]
last_index=0
first_index=0
for minibatch_index in range(n_test_batches):
   x=X_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   y=Y_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   # x=np.asarray(x[0])
   # y=np.asarray(y[0])
   # x=np.expand_dims(x,axis=0)
   # y=np.expand_dims(y,axis=0)

   last_index=first_index+sum([len(i) for i in y])
   n_list=N_list[first_index:last_index]
   first_index=last_index
   pred = model.predictions(x)
   # print("Prediction done....")
   if residual>0:
      if(minibatch_index==n_test_batches-1):
         pred = pred[0:(len(pred)-residual)]
         y=y[0:(len(y)-residual)]
         n_list=n_list[0:(len(n_list)-residual)]

#   du.write_predictions(params,pred,n_list)
   u.write_pred(pred,minibatch_index,G_list,params)
   loss=np.nanmean(np.abs(pred -y))
   (loss3d,l_list) =u.get_loss_bb(y,pred)
   loss_list=loss_list+l_list
   batch_loss += loss
   batch_loss3d += loss3d
batch_loss/=n_test_batches
batch_loss3d/=n_test_batches
s ='error %f, %f, %f'%(batch_loss,batch_loss3d,n_test_batches)
print (s)
pu.plot_histograms(loss_list)
pu.plot_error_frame(loss_list)
#pu.plot_cumsum(loss_list)
