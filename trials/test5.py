import os
import numpy

seq_id=0
s_list=[1 ,2, 3, 4, 7, 8, 9, 13, 14, 15, 18, 19, 20, 26, 27, 28]
s_list=[a-1 for a in s_list]
lst_act=['S9','S11','S1','S5','S6','S7','S8']

base_file="/mnt/Data1/hc/joints/"
new_file="/mnt/Data1/hc/joints16/"
# os.mkdir(new_file)
for actor in lst_act:
        print actor
        tmp_folder=base_file+actor+"/"
        tmp_folder2=new_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        os.mkdir(tmp_folder2)
        for sq in lst_sq:
            seq_id+=1
            tmp_folder=base_file+actor+"/"+sq+"/"
            tmp_folder2=new_file+actor+"/"+sq+"/"
            file_list=os.listdir(tmp_folder)
            os.mkdir(tmp_folder2)
            # print file_list
            for fl in file_list:
                p=tmp_folder+fl
                if not os.path.isfile(p):
                   continue
                with open(p, "rb") as f:
                  data=f.read().strip().split(' ')
                  y_d= numpy.asarray([float(val) for val in data])
                  tmp_y=y_d.reshape(32,3)
                  f_y=tmp_y[s_list]
                  f_y=f_y.flatten().T
                  numpy.savetxt(p.replace('joints','joints16'),f_y,fmt='%.4f',newline=' ')
