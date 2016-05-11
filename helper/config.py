import os
import utils
import platform

def get_params():
   global params
   params={}
   params['run_mode']=0 #0,full,1:resume, 2,X
   params["rn_id"]="real_cnn_highbias" #running id, model
   params["notes"]="real rcnn started with high bias values." #running id
   params["model"]="real_rcnn"#kccnr,dccnr
   params["optimizer"]="Adam" #1=classic kcnnr, 2=patch, 3=conv, 4 =single channcel
   params['seq_length']= 20
   params['validate']= 1
   params['mfile']= "xxx.p"

   params['batch_size']=10
   params['shufle_data']=0
   params['max_count']= 10

   #system settings
   wd=os.path.dirname(os.path.realpath(__file__))
   wd=os.path.dirname(wd)
   params['wd']=wd
   params['log_file']=wd+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   params["model_file"]=wd+"/cp/"
   params["data_dir"]="/mnt/Data2/DataFelix/hc/rnn/old_blanket/"

   # early-stopping parameters
   params['patience']= 10000  # look as this many examples regardless
   params['patience_increase']=2  # wait this much longer when a new best is
   params['improvement_threshold']=0.995  # a relative improvement of this much is

   # learning parameters
   params['momentum']=0.9    # the params for momentum
   params['lr']=0.0001
   params['learning_rate_decay']= 0.998
   params['squared_filter_length_limit']=15.0
   params['n_epochs']=25600
   params['n_hidden']= 512
   params['n_output']= 42

   if(platform.node()=="coskunh"):
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params['batch_size']=20
       params["WITH_GPU"]=False
       params['n_patch']= 1
       params['n_repeat']= 1
       params['n_hidden']= 128
       params['max_count']= 10000000

   if(platform.node()=="milletari-workstation"):
       params["data_dir"]="/mnt/Data2/DataFelix/dt/"
       params["caffe"]="/usr/local/caffe/python"
       params["WITH_GPU"]=True
       params['n_hidden']= 128
       params['max_count']=500

   if(platform.node()=="cmp-comp"):
       params['batch_size']=60
       params["n_procc"]=1
       params["WITH_GPU"]=True
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params["data_dir"]="/home/cmp/PycharmProjects/data/rnn/"
       params['n_hidden']= 128
       params['max_count']= 30

   #params['step_size']=[10]
   params['test_size']=0.20 #Test size
   params['val_size']=0.20 #val size
   params['test_freq']=100 #Test frequency
   return params

def update_params(params):
   params['log_file']=params["wd"]+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   return params