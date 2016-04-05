import glob
import struct
import os
import numpy
import h5py

def load_pose(params,only_test=0,only_pose=1,sindex=0):
   data_dir=params["data_dir"]
   max_count=params["max_count"]
   seq_length=params["seq_length"]
   # min_tr=0.000000
   # max_tr=8.190918
   # norm=2#numpy.linalg.norm(X_test)

   min_tr=0.000000
   max_tr=3
   norm=1#numpy.linalg.norm(X_test)

   istest=True
   get_flist=False
   F_list=[]
   G_list=[]
   (X_test,Y_test)= read_full_poseV2(data_dir,max_count,seq_length,sindex,istest)
   Y_test=Y_test/norm
   X_test=(X_test -min_tr) / (max_tr -min_tr)
   if only_test==1:
      return (X_test,Y_test,F_list,G_list)

   istest=False
   get_flist=False
   X_train,Y_train=read_full_poseV2(data_dir,max_count,seq_length,sindex,istest)
   if params['shufle_data']==1:
      X_train,Y_train=shuffle_in_unison_inplace(X_train,Y_train)

   #print("min: %f, max: %f "%(numpy.min(X_train),numpy.max(X_train)))

   Y_train=Y_train/norm
   X_train=(X_train -min_tr) / (max_tr -min_tr)

   return (X_train,Y_train,X_test,Y_test)


def read_full_poseV2(base_file,max_count,p_count,sindex,istest):
    if istest==0:
        lst_act=['felix','julia','ashley','dario','severine','huseyin','mona','philipp']
        lst_sq=['bodycalib','sleep','movementsh','getup','bodycalib','eatsO','eats','reads','sleeps','objectss','sleeph','getuph','getinbedh']
    else:
        lst_act=['leslie','meng']
        lst_sq=['bodycalib','sleep','movementsh','getup','bodycalib','eatsO','eats','reads','sleeps','objectss','sleeph','getuph','getinbedh']
    X_D=[]
    Y_D=[]
    for actor in lst_act:
        for sq in lst_sq:
            X_d=[]
            Y_d=[]
            for id in range(1,1801):
                fl=base_file+actor+"_"+str(sq)+"_"+str(id)+".txt"
                if not os.path.isfile(fl):
                    if id==1:
                        break;
                    continue
                with open(fl) as f:
                    lines = f.readlines()
                    data=lines[1].strip().split(' ')
                    y_d= [float(val) for val in data]

                    data=lines[0].strip().split(' ')
                    x_d= [float(val) for val in data]
                    X_d.append(x_d)
                    Y_d.append(y_d)
                    if len(X_d)==p_count and p_count>0:
                        X_D.append(X_d)
                        Y_D.append(Y_d)
                        X_d=[]
                        Y_d=[]
                    if len(X_D)>=max_count:
                        return (numpy.asarray(X_D),numpy.asarray(Y_D))

    return (numpy.asarray(X_D),numpy.asarray(Y_D))


def read_kinect_data(base_file,max_count,p_count,sindex):
    mat = h5py.File(base_file,mode='r',driver='core')
    max_train=max_count
    max_test=max_count/5
    X_D_train=[]
    Y_D_train=[]
    X_D_test=[]
    Y_D_test=[]
    X_d=[]
    Y_d=[]
    actor=""
    old_actor=""
    for id in range(189990):
        #actor
        rf=mat["imdb"]["meta"]["info"]["actor"].value[id][0]
        obj=mat[rf]
        actor=''.join(chr(i) for i in obj[:])

        rf=mat["imdb"]["meta"]["info"]["frame"].value[id][0]
        obj=mat[rf]
        frame_id= int(obj.value[0][0])

        if(frame_id==1):
            X_d=[]
            Y_d=[]

        if(old_actor==""):
            old_actor=actor
        if(actor!=old_actor):
            old_actor=actor
            X_d=[]
            Y_d=[]


        label=[x[0][0] for x in mat["imdb"]["images"]["labels"][id,:,:,:]]
        image=[x for x in mat["imdb"]["images"]["data"][id,:,:,:]]
        image=numpy.squeeze(image[0].flatten())
        X_d.append(image)
        Y_d.append(label)
        if len(X_d)==p_count and p_count>0:
               if(actor=="Felix" and len(X_D_test)<max_test):
                   X_D_test.append(X_d)
                   Y_D_test.append(Y_d)
               elif(len(X_D_test)<max_train):
                   X_D_train.append(X_d)
                   Y_D_train.append(Y_d)
               X_d=[]
               Y_d=[]
        print "id: %d, actor: %s, train: %d, test: %d" %(id,actor,len(X_D_train), len(X_D_test))

        if len(X_D_train)>=max_train and len(X_D_test)>=max_test:
            return (numpy.asarray(X_D_train),numpy.asarray(Y_D_train),numpy.asarray(X_D_test),numpy.asarray(Y_D_test))

    return (numpy.asarray(X_D_train),numpy.asarray(Y_D_train),numpy.asarray(X_D_test),numpy.asarray(Y_D_test))

def load_full_pose(base_file,max_count,istest,get_flist,p_count,sindex):
    if(istest):
        base_file=base_file+"test/test.txt"
    else:
        base_file=base_file+"train/train.txt"
    X_D=[]
    Y_D=[]
    p_index=0
    X_d=[]
    Y_d=[]
    F_L=[]
    G_L=[]
    old_f_sq_nm=-1
    with open(base_file) as f:
        lines = f.readlines()
        fl_len=len(lines)/4
        for nm in range(sindex,fl_len):
            fname=lines[nm*4+0].replace("\n","")
            frame=lines[nm*4+1].replace("\n","")
            lname=lines[nm*4+2].replace("\n","")
            label=lines[nm*4+3].replace("\n","")
            f_sq_nm=int(fname.split("_")[1].replace("seq",""))

            # l_sq_nm=int(lname.split("_")[1].replace("seq"))
            # f_nm=int(lname.split("_")[2].replace("frame"))
            if(old_f_sq_nm==-1):
                old_f_sq_nm=f_sq_nm
            # if(f_sq_nm==13 or f_sq_nm==11):
            #      continue

            if(f_sq_nm != old_f_sq_nm):
                if p_count==-1 and len(X_d)>0:
                    X_D.append(X_d)
                    Y_D.append(Y_d)
                    p_index=p_index+1
                old_f_sq_nm=f_sq_nm
                X_d=[]
                Y_d=[]

            if(get_flist):
                F_L.append(fname)
                G_L.append(lname)

            data=label.strip().split(' ')
            y_d= [float(val) for val in data]
            Y_d.append(numpy.asarray(y_d))

            data=frame.strip().split(' ')
            x_d = [float(val) for val in data]
            X_d.append(numpy.asarray(x_d))

            if len(X_d)==p_count and p_count>0:
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

            if len(X_D)==max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)

        if p_count==-1 and len(X_d)>0:
                    X_D.append(X_d)
                    Y_D.append(Y_d)

    return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)

def load_test_poseV2(base_file,max_count,p_count,sindex):
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
    F_L=[]
    G_L=[]
    for sq in sql:
        X_d=[]
        Y_d=[]
        for fm in range(sindex,1801,1):
           if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)
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
           F_L.append(fl)
           G_L.append(gl)
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


    return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)

def load_train_poseV2(base_file,max_count,p_count,sindex):
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
        for fm in range(sindex,1801,1):
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
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

    return (numpy.asarray(X_D),numpy.asarray(Y_D))

def load_full_poseV3(is_train,base_file,max_count,p_count,sindex):
    if(is_train==True):
        base_file=base_file+"train/"
    else:
        base_file=base_file+"test/"
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
        for fm in range(sindex,1801,1):
            if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D))
            fl=base_file+ft_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
            gl=base_file+gt_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
            if not os.path.isfile(fl):
               continue
            Y_d.append(gl)
            X_d.append(fl)

            if len(X_d)>p_count:
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

    return (numpy.asarray(X_D),numpy.asarray(Y_D))

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
    F_L=[]
    G_L=[]
    for sq in sql:
        X_d=[]
        Y_d=[]
        for fm in range(0,1801,1):
           if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)
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
           F_L.append(fl)
           G_L.append(gl)
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


    return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)

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