import numpy as np
import helper.config as config
import model.model_provider as model_provider
import helper.dt_utils as du
import helper.utils as u

params= config.get_params()
params["model"]="lstm"
model_file="lstm_model_test_0.p"
only_test=1
params['batch_size']=64

(X_test,Y_test)=du.load_pose(params,only_test)
batch_size=params['batch_size']

n_test_batches = len(X_test)
n_test_batches /= batch_size

print ("Model loading started")
model= model_provider.get_model_pretrained(params,model_file)
print("Prediction started")
batch_loss = 0.
batch_loss3d = 0.
for minibatch_index in range(n_test_batches):
   x=X_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   y=Y_test[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
   pred = model.predictions(x)
   loss=np.mean(np.abs(pred - y))
   loss3d =u.get_loss(y,pred)
   batch_loss += loss
   batch_loss3d += loss3d
batch_loss/=n_test_batches
batch_loss3d/=n_test_batches
s ='error %f, %f '%(batch_loss,batch_loss3d)
print (s)
