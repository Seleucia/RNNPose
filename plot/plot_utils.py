import numpy as np

def plot_histograms(loss):
   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(111)
   x = loss
   bins = 50
   ax.hist(x,bins=bins,color='green')
   plt.show()


def plot_cumsum(loss):
   import matplotlib.pyplot as plt

   # Create some test data
   x  = loss
   CY = np.cumsum(x)
   plt.plot(x,CY,'r--')
   plt.show()
