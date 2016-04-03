import numpy as np
import os
import scipy.io
import datetime
import collections
from PIL import Image


def read_kinect_data():
    filename=""
    mat = scipy.io.loadmat(filename)
    lines_date=mat['mdata']['fdate'][0][0][0]
    lines_depth=mat['mdata']['depth'][0][0][0]
    line_depth=lines_depth[1:-1]



read_kinect_data()