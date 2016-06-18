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


def plot_percentile(loss):
   print 'started'

   loss=np.asarray(loss)*1000
   loss=loss.flatten()
   print loss.shape
   mx=np.max(loss)
   mn=np.min(loss)
   diff=(mx-mn)/100.
   lst=[]
   lst.append(mn)
   vl=mn
   while vl<mx:
      vl=vl+diff
      lst.append(vl)

   perc=[]
   # print lst
   for l in lst:
      rt=float(len(loss[np.where(loss<=l)]))/float(len(loss))
      perc.append(rt)

   # print perc
   import matplotlib.pyplot as plt
   fig = plt.figure()
   plt.plot(lst,perc)
   # fig.suptitle('Accuracy with threshold(mm)')
   plt.xlabel('Threshold(mm)')
   plt.ylabel('Accuracy')
   # plt.title('Te')
   fig.savefig('test.jpg')
   # Set tick locations and labels
   # plt.xticks(lst)

   plt.show()






def plot_cumsum(loss):
   import matplotlib.pyplot as plt

   # Create some test data
   x  = loss
   CY = np.cumsum(x)
   plt.plot(x,CY,'r--')
   plt.show()