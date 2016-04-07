from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy
import numpy as np


def plot_raw_y(data_y,fig_name,params):
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
   color=['red', 'green']
   plt.gca().set_color_cycle(['red'])
   legends=[]
   legends.append("GT data")

   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')
   x =data_y[:,0]
   y =data_y[:,1]
   z =data_y[:,2]
   ax.plot(x, y, z, color[0])

   ax.legend(legends,loc='upper left')
   plt.savefig("predictions/"+fig_name)
   plt.show()

def plot_orijinal_y(y_train_delta_gt,y_test_delta_gt,y_val_delta_gt):
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
   plt.gca().set_color_cycle(['red','green','blue'])
   legends=[]
   legends.append("Train data")
   legends.append("Test data")
   legends.append("Validation data")

   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')
   i=0
   x =y_train_delta_gt[:,0]
   y =y_train_delta_gt[:,1]
   z =y_train_delta_gt[:,2]
   ax.plot(x, y, z,'red')

   x =y_test_delta_gt[:,0]
   y =y_test_delta_gt[:,1]
   z =y_test_delta_gt[:,2]
   ax.plot(x, y, z,'green')

   x =y_val_delta_gt[:,0]
   y =y_val_delta_gt[:,1]
   z =y_val_delta_gt[:,2]
   ax.plot(x, y, z,'blue')

   ax.legend(legends,loc='upper left')
   plt.show()

def plot_y(data_y,fig_name):
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
   color=['red', 'green']
   plt.gca().set_color_cycle(['red', 'green'])
   legends=[]
   legends.append("GT data")
   legends.append("Est data")

   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')
   i=0
   for d in data_y:
       x =d[:,0]
       y =d[:,1]
       z =d[:,2]
       ax.plot(x, y, z, color[i])
       i +=1


   ax.legend(legends,loc='upper left')
   plt.savefig(fig_name)
   plt.show()

def plot_err(err,fig_name):
   import matplotlib.pyplot as plt
   f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
   fr = range(len(err))
   x = err[:, 0]
   ax1.plot(fr, x)
   plt.grid()

   y = err[:, 1]
   ax2.plot(fr, y)
   plt.grid()

   z = err[:, 2]
   ax3.plot(fr, z)
   plt.grid()
   #f.subplots_adjust(hspace=0)
   plt.savefig(fig_name)
   #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
   plt.show()

def plot_val(list_val,fig_name,tt='Validation error change with epoch',lim=0):
   import matplotlib.pyplot as plt
   nd=numpy.array([[epoch,error] for (epoch,error) in list_val])

   y = nd[nd[:,0]>lim][:,0] #all error
   x =nd[nd[:,0]>lim][:,1] #all error
   plt.plot(y, x)

   plt.xlabel('epoch (x)')
   plt.ylabel('error (t)')
   plt.title(tt)
   plt.grid(True)
   #f.subplots_adjust(hspace=0)
   plt.savefig(fig_name)
   #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
   plt.show()

def plot_val_train(list_train,fig_name,epoch):
   import matplotlib.pyplot as plt

   if(epoch==-1):
       nd=numpy.array([[int(b), int(c), d] for (b, c, d) in list_train]) #all error
       idx=map(int,nd[:,0])
       err=nd[:,2]
       y=numpy.bincount(idx, err)[1:len(idx)+1] / np.bincount(idx)[1:len(idx)+1]
       x =[x+1 for x in range(len(y))]
       plt.title('Train Error change with epoch')
       plt.xlabel('epoch (x)')
   else:
       y = numpy.array([[b, c, d] for (b, c, d) in list_train if b==epoch ])[:,2] #all error
       x =numpy.array([[b, c, d] for (b, c, d) in list_train if b==epoch ])[:,1]  #all error
       plt.title('Train Error change with minibatch')
       plt.xlabel('minibatc (x)')


   plt.plot(x, y)
   plt.ylabel('error (y)')
   plt.grid(True)
   #f.subplots_adjust(hspace=0)
   plt.savefig(fig_name)
   #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
   plt.show()

def plot_ms(data,dir,name):
   import matplotlib.pyplot as plt
   y = data
   x =range(len(data))
   plt.plot(x,y)

   plt.xlabel('x')
   plt.ylabel('y')
   plt.title(name)
   plt.savefig(dir.replace(name+"/","")+name)
   plt.grid(True)
   plt.show()



def plot_patch(pred_mat):
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt

   for i in range(len(pred_mat)):
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for k in range(len(pred_mat[i])):
         xs = pred_mat[i][k][0]
         ys = pred_mat[i][k][1]
         zs = pred_mat[i][k][2]
         ax.scatter(xs, ys, zs)

      ax.set_xlabel('X Label')
      ax.set_ylabel('Y Label')
      ax.set_zlabel('Z Label')
      #plt.show()
      plt.savefig("/home/coskun/Desktop/sil/1/"+str(i)+".jpg")
      print "Look image"

