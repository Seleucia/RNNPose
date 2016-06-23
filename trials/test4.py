import os
import numpy
from PIL import Image
from glob import glob

os.environ["CDF_LIB"]='/home/coskun/sftpkg/cdf36_1-dist/lib'
from spacepy import pycdf
# cdf1 = pycdf.CDF('/mnt/Data2/DataFelix/3.6m/TOF_S1/TOF/Walking.cdf')
pp='/home/coskun/Downloads/S1/MyPoseFeatures/D3_Positions_mono/Walking.60457274.cdf'
cdf = pycdf.CDF(pp)
cdf_dat = cdf.copy()
print "ok"
