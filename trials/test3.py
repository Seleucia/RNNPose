import os
import numpy
from PIL import Image
from glob import glob

os.environ["CDF_LIB"]='/home/coskun/sftpkg/cdf36_1-dist/lib'
from spacepy import pycdf
# cdf1 = pycdf.CDF('/mnt/Data2/DataFelix/3.6m/TOF_S1/TOF/Walking.cdf')
pp='//mnt/Data1/hc/cdf2/S11/MyPoseFeatures/D3_Positions/Walking.cdf'


paths = os.listdir('/mnt/Data1/hc/cdf2')

for sb in paths:
    p_folder='/mnt/Data1/hc/cdf2/'+sb+'/MyPoseFeatures/D3_Positions/'
    new_path='/mnt/Data1/hc/joints/'+sb+'/'
    if not os.path.exists(new_path):
            os.mkdir(new_path)
    files=os.listdir(p_folder)
    for f in files:
        pp=p_folder+f
        cdf = pycdf.CDF(pp)
        cdf_dat = cdf.copy()
        # cdf_dat1 = cdf1.copy()
        # fname=os.path.basename(pp)
        new_filder=new_path+f
        if not os.path.exists(new_filder):
            os.mkdir(new_filder)

        mat=numpy.asarray(cdf_dat['Pose'])[0]
        counter=1
        for r in mat:
            new_file=new_filder+'/'+str(counter)+'.txt'
            numpy.savetxt(new_file, r, delimiter=' ',newline=" ",fmt="%.4f")
            counter+=1
            # print r
        # im=(mat[200]/5000)*255
