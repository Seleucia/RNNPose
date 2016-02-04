import numpy as np

def plot_histograms(loss):
   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(111)
   x = loss
   bins = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,30,50)
   ax.hist(x,bins=bins,color='green',alpha=0.8)
   plt.show()
