import numpy as np
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import plot.plot_utils as pu
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
params= config.get_params()
params["model"]="cnn"
params['mfile']= "cnn_S9-11_tr_cnnV2_9_0.475661493513_best.p"
# params["data_dir"]="/mnt/Data2/DataFelix/dt/"
# params["data_dir"]="/home/coskun/PycharmProjects/data/rnn/180k/"
#0, error 0.087360, 0.177598, 20.323595
#error 0.078438, 0.161955, 16.453038
#VAL--> epoch 21 | error 0.086701, 0.179906
rng = RandomStreams(seed=1234)
only_test=1
is_train=0
params['seq_length']= 20
params['batch_size']=100
batch_size=params['batch_size']

u.prep_pred_file(params)

sindex=0
(X_test,Y_test,F_list_test,G_list_test,S_Test_list)=du.load_pose(params,only_test=1,sindex=0)

n_test = len(F_list_test)
residual=n_test%batch_size
#residual=0
if residual>0:
   residual=batch_size-residual
   # S_Test_list_list=S_Test_list.tolist()
   f=F_list_test[-1]
   g=G_list_test[-1]
   # s=S_Test_list_list[-1]
   for i in range(residual):
      F_list_test.append(f)
      G_list_test.append(g)
      # S_Test_list_list.append(s)
   # S_Test_list=np.asarray(S_Test_list_list)
   n_test = len(F_list_test)

# n_test_batches =n_test/ batch_size
n_test_batches = len(F_list_test)
n_test_batches /= batch_size

print("Test sample size: %i, Batch size: %i, test batch size: %i"%(len(F_list_test),batch_size,n_test_batches))
print ("Model loading started")
model= model_provider.get_model_pretrained(params,rng)
print("Prediction started")
batch_loss = 0.
batch_loss3d = 0.
batch_bb_loss = 0.
loss_list=[]
last_index=0
first_index=0
sq_loss_lst=[]
for minibatch_index in range(n_test_batches):
   x_lst=F_list_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
   y_lst=G_list_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] #60*20*1024
   x,y=du.load_batch(params,x_lst,y_lst)

   last_index=first_index+sum([len(i) for i in y])
   first_index=last_index
   if(params["model"]=="blstmnp"):
      x_b=np.asarray(map(np.flipud,x))
      pred = model.predictions(x,x_b,is_train)
   else:
      pred = model.predictions(x,is_train)
   # print("Prediction done....")
   if residual>0:
      if(minibatch_index==n_test_batches-1):
         pred = pred[0:(len(pred)-residual)]
         y=y[0:(len(y)-residual)]

         # n_list=n_list[0:(len(n_list)-residual)]
   dif=(pred-y)*(5./6.)
   pred=pred-dif

#   du.write_predictions(params,pred,n_list)
   #u.write_pred(pred,minibatch_index,G_list,params)
   loss=np.nanmean(np.abs(pred -y))
   (loss3d,l_list) =u.get_loss_pred(params,y,pred)
   #print(s_list)
   loss_list=loss_list+l_list
   batch_loss += loss
   batch_loss3d += loss3d
sq_loss_lst=np.nanmean(sq_loss_lst,axis=0)
batch_loss/=n_test_batches
batch_loss3d/=n_test_batches
print "============================================================================"
# print sq_loss_lst
s ='error %f, %f, %f,%f,%f'%(batch_loss,batch_loss3d,np.mean(loss_list),n_test_batches,len(loss_list))
print (s)
# pu.plot_histograms(loss_list)
# pu.plot_error_frame(loss_list)
pu.plot_percentile(loss_list)
