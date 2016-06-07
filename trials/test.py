import numpy
from PIL import Image
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')


def show_image():
    X_d=[]
    Y_d=[]
    base_file="/mnt/Data1/hc/img/S11/Directions 1.cdf/"
    fl=base_file+'100.txt'
    gl=base_file.replace('img','joints')+'1.txt'
    if not os.path.isfile(fl):
        print 'file not found %s'%(fl)
        return
    with open(gl, "rb") as f:
      data=f.read().strip().split(' ')
      y_d= [float(val) for val in data]
      # if(numpy.isnan(numpy.sum(y_d))):
      #     continue;
      Y_d.append(numpy.asarray(y_d)/1000)

    with open(fl, "rb") as f:
        data=f.read().strip().split(' ')
        data=data[0].split('\n')
        x_d = [float(val) for val in data]
        Z=numpy.reshape(numpy.asarray(x_d),newshape=(144,176))
        # Z[Z > 5] = 0
        # Z[Z < 3.5] = 0

        # #Create X and Y data
        # xmin=ymin=0
        # xmax=144./100.
        # ymax=176./100.
        # xstep=1./100.
        # ystep=1./100.
        # x = np.arange(xmin, xmax, xstep)
        # y = np.arange(ymin, ymax, ystep)
        # X, Y = np.meshgrid(x, y)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, antialiased=True)
        # #Show the plot
        # plt.show()

        img = Image.fromarray(Z,'LA')
        img.save('my3.png')

show_image()