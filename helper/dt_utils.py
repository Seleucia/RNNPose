import tables
import struct
import os
import numpy

def load_pose(params):
   data_dir=params["data_dir"]
   max_count=params["max_count"]
   seq_length=params["seq_length"]
   X_test,Y_test=load_test_pose(data_dir,max_count,seq_length)
   if params['shufle_data']==1:
      X_test,Y_test=shuffle_in_unison_inplace(X_test,Y_test)
   X_train,Y_train=load_train_pose(data_dir,max_count,seq_length)
   return (X_train,Y_train,X_test,Y_test)

def load_test_pose(base_file,max_count,p_count):
   base_file=base_file+"test/"
   X_D=[]
   Y_D=[]
   p_index=0
   X_d=[]
   Y_d=[]
   for vw in range(1,12,1):
       for sq in range(1,4,1):
           for sb in range(1,2,1):
               for fm in range(1,1200,1):
                   if len(X_D)>max_count:
                       return (numpy.asarray(X_D),numpy.asarray(Y_D))
                   fl=base_file+"view"+str(vw)+"_seq"+str(sq)+"_subj"+str(sb)+"_frame"+str(fm)+".txt"
                   gl=base_file+"GTjoints_view"+str(vw)+"_seq"+str(sq)+"_subj"+str(sb)+"_frame"+str(fm)+".txt"
                   if not os.path.isfile(fl):
                       continue
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

   return (numpy.asarray(X_D),numpy.asarray(Y_D))

def load_train_pose(base_file,max_count,p_count):
   base_file=base_file+"train/"
   X_D=[]
   Y_D=[]
   p_index=0
   X_d=[]
   Y_d=[]
   for vw in range(1,12,1):
       for sq in range(4,8,1):
           for sb in range(2,9,1):
               for fm in range(1,300,1):
                   if len(X_D)>max_count:
                       return (numpy.asarray(X_D),numpy.asarray(Y_D))
                   fl=base_file+"view"+str(vw)+"_seq"+str(sq)+"_subj"+str(sb)+"_frame"+str(fm)+".txt"
                   gl=base_file+"GTjoints_view"+str(vw)+"_seq"+str(sq)+"_subj"+str(sb)+"_frame"+str(fm)+".txt"
                   if not os.path.isfile(fl):
                       continue
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

   return (numpy.asarray(X_D),numpy.asarray(Y_D))

def shuffle_in_unison_inplace(a, b):
   assert len(a) == len(b)
   p = numpy.random.permutation(len(a))
   return a[p], b[p]