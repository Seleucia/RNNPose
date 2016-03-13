import glob
import struct
import os
import numpy
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

def load_pose(params,only_test=0,only_pose=1):
   data_dir=params["data_dir"]
   max_count=params["max_count"]
   seq_length=params["seq_length"]

   X_train,Y_train=load_train_pose(data_dir,max_count,seq_length)

   X_test,Y_test,N_list=load_test_pose(data_dir,max_count,seq_length)
   if only_test==1:
      return (X_test,Y_test,N_list)

   if params['shufle_data']==1:
      X_train,Y_train=shuffle_in_unison_inplace(X_train,Y_train)

   #norm=numpy.linalg.norm(X_test)
   norm=2#numpy.linalg.norm(X_test)

   Y_test=Y_test/norm
   Y_train=Y_train/norm

   #norm=numpy.linalg.norm(X_test)
   X_test=(X_test -numpy.min(X_test)) / (numpy.max(X_test) -numpy.min(X_test))
   X_train=(X_train -numpy.min(X_train)) / (numpy.max(X_train) -numpy.min(X_train))

   return (X_train,Y_train,X_test,Y_test)


def load_test_pose_v1(base_file,max_count,p_count):
    base_file=base_file+"test/"
    ft_prefix="feature_"
    gt_prefix="label_"
    #feature_seq13_frame2.txt
    #featureVec1024_calib_Huseyin_view360_frame2,featureVec1024_getUp_Huseyin_view360_frame380
    #sbl=["Huseyin"]#range(1,1,1):
    sql=range(13,15,1);
    X_D=[]
    Y_D=[]
    p_index=0
    X_d=[]
    Y_d=[]
    N_L=[]
    for sq in sql:
       for fm in range(1,1800,1):
           if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D),N_L)
           #fl=base_file+ft_prefix+str(vw)+"_seq"+str(sq)+"_subj"+str(sb)+"_frame"+str(fm)+".txt"
           #gl=base_file+gt_prefix+str(vw)+"_seq"+str(sq)+"_subj"+str(sb)+"_frame"+str(fm)+".txt"
           fl=base_file+ft_prefix+"_seq"+str(sq)+"_frame"+str(fm)+".txt"
           gl=base_file+gt_prefix+"_seq"+str(sq)+"_frame"+str(fm)+".txt"
           if not os.path.isfile(fl):
               continue
           N_L.append(fl)
           with open(fl, "rb") as f:
               data = f.read(4)
               x_d=[]
               while data != "":
                   floatVal = struct.unpack('f', data)
                   x_d.append(floatVal[0])
                   data = f.read(4)
               X_d.append(numpy.asarray(x_d))
           with open(gl, "rb") as f:
               data = f.read(4)
               y_d=[]
               while data != "":
                   floatVal = struct.unpack('f', data)
                   y_d.append(floatVal[0])
                   data = f.read(4)
               Y_d.append(numpy.asarray(y_d))
           if len(X_d)>p_count:
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

    return (numpy.asarray(X_D),numpy.asarray(Y_D),N_L)

def load_test_pose(base_file,max_count,p_count):
    base_file=base_file+"test/"
    ft_prefix="feature_"
    gt_prefix="label_"
    #feature_seq13_frame2.txt
    #featureVec1024_calib_Huseyin_view360_frame2,featureVec1024_getUp_Huseyin_view360_frame380
    #sbl=["Huseyin"]#range(1,1,1):
    sql=range(13,15,1);
    X_D=[]
    Y_D=[]
    p_index=0
    X_d=[]
    Y_d=[]
    N_L=[]
    for sq in sql:
        X_d=[]
        Y_d=[]
        for fm in range(0,1801,1):
           if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D),N_L)
           fl=base_file+ft_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
           gl=base_file+gt_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
           if not os.path.isfile(gl):
               continue
           with open(gl, "rb") as f:
              data=f.read().strip().split(' ')
              y_d= [float(val) for val in data]
              # if(numpy.isnan(numpy.sum(y_d))):
              #     continue;
              Y_d.append(numpy.asarray(y_d))
           with open(fl, "rb") as f:
               data=f.read().strip().split(' ')
               x_d = [float(val) for val in data]
               X_d.append(numpy.asarray(x_d))
           N_L.append(fl)
           if len(X_d)>p_count and p_count>0:
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

        if p_count==-1 and len(X_d)>0:
          X_D.append(X_d)
          Y_D.append(Y_d)
          p_index=p_index+1


    return (numpy.asarray(X_D),numpy.asarray(Y_D),N_L)

def load_train_pose(base_file,max_count,p_count):
    base_file=base_file+"train/"
    ft_prefix="feature_"
    gt_prefix="label_"
    #feature_seq13_frame2.txt
    #featureVec1024_calib_Huseyin_view360_frame2,featureVec1024_getUp_Huseyin_view360_frame380
    #sbl=["Huseyin"]#range(1,1,1):
    sql=range(1,14,1);
    X_D=[]
    Y_D=[]
    p_index=0
    X_d=[]
    Y_d=[]
    for sq in sql:
        X_d=[]
        Y_d=[]
        for fm in range(1,1801,1):
            if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D))
            fl=base_file+ft_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
            gl=base_file+gt_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
            if not os.path.isfile(fl):
               continue

            with open(gl, "rb") as f:
              data=f.read().strip().split(' ')
              y_d= [float(val) for val in data]
              # if(numpy.isnan(numpy.sum(y_d))):
              #     continue;
              Y_d.append(numpy.asarray(y_d))

            with open(fl, "rb") as f:
               data=f.read().strip().split(' ')
               x_d = [float(val) for val in data]
               X_d.append(numpy.asarray(x_d))

            if len(X_d)>p_count:
               print fm
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

    return (numpy.asarray(X_D),numpy.asarray(Y_D))

def load_test_bboxes(params,max_count):
   data_dir=params["data_dir"]
   base_file=data_dir+"bboxes_test_SynthData_07_31/"
   #bbox_view1_seq1_subj1_frame1
   X_d=[]
   for vw in range(1,12,1):
       for sq in range(1,4,1):
           for sb in range(1,2,1):
               for fm in range(1,1200,1):
                   if len(X_d)==max_count:
                       return (numpy.asarray(X_d))
                   fl=base_file+"bbox_view"+str(vw)+"_seq"+str(sq)+"_subj"+str(sb)+"_frame"+str(fm)+".txt"
                   if not os.path.isfile(fl):
                       continue
                   with open(fl, "rb") as f:
                       data = f.read(4)
                       x_d=[]
                       while data != "":
                           floatVal = struct.unpack('f', data)
                           x_d.append(floatVal[0])
                           data = f.read(4)
                       x_d=((numpy.abs(numpy.asarray(x_d).reshape(3,2)[:,0].T- numpy.asarray(x_d).reshape(3,2)[:,1].T))/10)
                       X_d.append(x_d)
   return (numpy.asarray(X_d))

def get_folder_name_list(params):
   data_dir=params["data_dir"]
   max_count=params["max_count"]
   base_file=data_dir+"test/"

   return glob.glob("/home/adam/*.txt")

def write_predictions(params,pred,N_list):
   wd=params["wd"]
   base_dir=wd+"/pred/res/"
   counter=0
   for b in pred:
      for i in b:
         path=base_dir+params["model"]+"_"+os.path.split(N_list[counter])[1].replace(".txt","")
         numpy.save(path, i)
         counter=counter+1

def shuffle_in_unison_inplace(a, b):
   assert len(a) == len(b)
   p = numpy.random.permutation(len(a))
   return a[p], b[p]