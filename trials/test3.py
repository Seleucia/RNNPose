import numpy
import h5py


def read_kinect_data(base_file,max_count,p_count,sindex):
    mat = h5py.File(base_file)
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
    for id in range(1000):
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

        if len(X_D_train)==max_train and len(X_D_test)==max_test:
            return (numpy.asarray(X_D_train),numpy.asarray(Y_D_train),numpy.asarray(X_D_test),numpy.asarray(Y_D_test))

    return (numpy.asarray(X_D_train),numpy.asarray(Y_D_train),numpy.asarray(X_D_test),numpy.asarray(Y_D_test))


base_file="/mnt/Data2/DataFelix/imdb_blanket_10_subjects.mat"
p_count=20
sindex=0
max_count=100
(X_train,Y_train,X_test,Y_test)=read_kinect_data(base_file=base_file,max_count=max_count,p_count=p_count,sindex=sindex)
print "done...."