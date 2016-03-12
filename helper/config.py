from model.lstm import  lstm
import os
import utils
import platform

def get_params():
   global params
   params={}
   params['run_mode']=1 #0,full,1:only for check, 2: very small ds, 3:only ICL data
   params["rn_id"]="model_test" #running id, model
   params["notes"]="Im testing my data loading approach" #running id
   params["model"]="blstmnp"#kccnr,dccnr
   params["optimizer"]="RMSprop" #1=classic kcnnr, 2=patch, 3=conv, 4 =single channcel
   params['seq_length']= 20
   params['validate']= 1
   params['resume']= 0
   params['mfile']= "gru_model_test_x.p"

   params['batch_size']=62
   params['shufle_data']=1
   params['max_count']= 300

   #system settings
   wd=os.path.dirname(os.path.realpath(__file__))
   wd=os.path.dirname(wd)
   params['wd']=wd
   params['log_file']=wd+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   params["model_file"]=wd+"/cp/"
   params["data_dir"]="/home/coskun/PycharmProjects/data/pose/30k/"


   # early-stopping parameters
   params['patience']= 10000  # look as this many examples regardless
   params['patience_increase']=2  # wait this much longer when a new best is
   params['improvement_threshold']=0.995  # a relative improvement of this much is

   # learning parameters
   params['momentum']=0.9    # the params for momentum
   params['initial_learning_rate']=0.0001
   params['learning_rate_decay']= 0.998
   params['squared_filter_length_limit']=15.0
   params['n_epochs']=25600
   params['n_hidden']= 512

   if(platform.node()=="coskunh"):
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params['batch_size']=20
       params["WITH_GPU"]=False
       params['n_patch']= 1
       params['n_repeat']= 1
       params['n_hidden']= 128
       params['max_count']= 100000

   if(platform.node()=="milletari-workstation"):
       params["caffe"]="/usr/local/caffe/python"
       params["WITH_GPU"]=True
       params['n_hidden']= 256
       params['max_count']= 30000000

   if(platform.node()=="cmp-comp"):
       params['batch_size']=62
       params["n_procc"]=1
       params["WITH_GPU"]=True
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params["data_dir"]="/home/cmp/PycharmProjects/data/rnn/"
       params['n_hidden']= 128
       params['max_count']= 3000000

   #params['step_size']=[10]
   params['test_size']=0.20 #Test size
   params['val_size']=0.20 #val size
   params['test_freq']=100 #Test frequency
   return params

def update_params(params):
   params['log_file']=params["wd"]+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   return params