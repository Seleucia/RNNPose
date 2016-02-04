import numpy as np

def plot_histograms(loss):
   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(111)
   x = loss
   bins = 50
   ax.hist(x,bins=bins,color='green')
   plt.title('Test in 3d for per frame')
   plt.xlabel('error (x)')
   plt.ylabel('number of frame (y)')
   plt.show()

def plot_error_frame(loss):
   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(111)
   y = loss
   x = range(len(loss))
   plt.plot(x,y,'r--')
   plt.title('Test error in 3d for per frame')
   plt.xlabel('frame id')
   plt.ylabel('error')
   plt.show()



def plot_cumsum(loss):
   import matplotlib.pyplot as plt

   # Create some test data
   x  = loss
   CY = np.cumsum(x)
   plt.plot(x,CY,'r--')
   plt.show()