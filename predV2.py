import numpy as np
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u
import plot.plot_utils as pu
params= config.get_params()
params["model"]="erd"
params['mfile']= "erd_model_test_0.p"
#0, error 0.087360, 0.177598, 20.323595
#error 0.078438, 0.161955, 16.453038
#VAL--> epoch 21 | error 0.086701, 0.179906

only_test=1
params['seq_length']= 19
params['batch_size']=30
batch_size=params['batch_size']
seq_length=params["seq_length"]



u.prep_pred_file(params)

sindex=0
data=[]
for sindex in range(0,seq_length,1):
   (X_test,Y_test,N_list,G_list)=du.load_pose(params,only_test=1,sindex=sindex)
   data.append((X_test,Y_test,N_list,G_list))

print ("Data loaded...")
full_lost=[]

model= model_provider.get_model_pretrained(params)
print ("Model Loaded")
for sp in range(seq_length):
   (X_test,Y_test,N_list,G_list)=data[sp]
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
      #u.write_pred(pred,minibatch_index,G_list,params)
      loss=np.nanmean(np.abs(pred -y))*2
      (loss3d,l_list) =u.get_loss_bb(y,pred)
      loss_list=loss_list+l_list
      batch_loss += loss
      batch_loss3d += loss3d

   batch_loss/=n_test_batches
   batch_loss3d/=n_test_batches
   full_lost.append(loss_list)
   s ='error %f, %f, %f'%(batch_loss,batch_loss3d,n_test_batches)
   print (s)

all=1
final_loss=[0]*(len(full_lost[0])+seq_length)
for sp in range(seq_length):
   ls=full_lost[sp]
   print len(ls)
   if(all==0):
      if(sp==0):
         final_loss[0:seq_length]=ls[0:seq_length]
      for index in range(seq_length,len(ls),seq_length):
         f_index=sp+index
         final_loss[f_index-3]=ls[index-3]
   else:
      print(sp)
      for index in range(0,len(ls)):
         f_index=sp+index
         final_loss[f_index]=final_loss[f_index]+ls[index]



final_loss=final_loss[seq_length:len(final_loss)-seq_length]
print len(final_loss)
final_loss=[l/seq_length for l in final_loss if l != -1]
print len(final_loss)
final_mean=np.mean(final_loss)

s ='Final mean error %f'%(final_mean)
print(s)
pu.plot_histograms(final_loss)
pu.plot_error_frame(final_loss)

   #pu.plot_cumsum(loss_list)
