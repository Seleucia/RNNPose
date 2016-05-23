import os
import numpy
from PIL import Image

os.environ["CDF_LIB"]='/home/coskun/sftpkg/cdf36_1-dist/lib'
from spacepy import pycdf
cdf1 = pycdf.CDF('/mnt/Data2/DataFelix/3.6m/TOF_S1/TOF/Walking.cdf')
cdf = pycdf.CDF('/mnt/Data2/DataFelix/3.6m/TOF_S1/S1/MyPoseFeatures/D3_Positions/Walking.cdf')
print(cdf)
cdf_dat = cdf.copy()
cdf_dat1 = cdf1.copy()
mat=numpy.asarray(cdf_dat1['IntensityFrames'])[0].reshape(1544,176,144)
im=(mat[200]/5000)*255
print im
img = Image.fromarray(im,mode='LA')

Image.fromarray(im).show(title='tt')

img.save("tt.png")