import tables
import struct
import os
import numpy

def laod_pose():
   base_file="/home/coskun/PycharmProjects/data/rnn/"
   max_count=100
   p_count=20
   X_test,Y_test=laod_test_pose(base_file,max_count,p_count)
   X_train,Y_train=laod_train_pose(base_file,max_count,p_count)
   return (X_train,Y_train,X_test,Y_test)

def laod_test_pose(base_file,max_count,p_count):
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

def laod_train_pose(base_file,max_count,p_count):
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