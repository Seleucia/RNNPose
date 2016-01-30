import numpy
import theano
import theano.tensor as T
import os
import glob
from PIL import Image
import datetime
import numpy as np
from random import randint
from theano import shared
dtype = T.config.floatX

def numpy_floatX(data):
    return numpy.asarray(data, dtype=T.config.floatX)

def init_weight(shape, name, sample='uni', seed=None):
    rng = np.random.RandomState(seed)

    if sample == 'unishape':
        values = rng.uniform(
            low=-np.sqrt(6. / (shape[0] + shape[1])),
            high=np.sqrt(6. / (shape[0] + shape[1])),
            size=shape).astype(dtype)

    elif sample == 'svd':
        values = rng.uniform(low=-1., high=1., size=shape).astype(dtype)
        _, svs, _ = np.linalg.svd(values)
        # svs[0] is the largest singular value
        values = values / svs[0]

    elif sample == 'uni':
        values = rng.uniform(low=-0.1, high=0.1, size=shape).astype(dtype)

    elif sample == 'zero':
        values = np.zeros(shape=shape, dtype=dtype)

    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)

    return shared(values, name=name, borrow=True)



def init_W_b(W, b, rng, n_in, n_out):
    # for a discussion of the initialization, see
    # https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa
    if W is None:
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6./(n_in + n_out)),
                high=numpy.sqrt(6./(n_in + n_out)),
                size=(n_in, n_out)
                ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)

    # init biases to positive values, so we should be initially in the linear regime of the linear rectified function
    if b is None:
        b_values = numpy.ones((n_out,), dtype=theano.config.floatX) * numpy.cast[theano.config.floatX](0.01)
        b = theano.shared(value=b_values, name='b', borrow=True)
    return W, b


def init_CNNW_b(W, b, rng, n_in, n_out,fshape):
    # for a discussion of the initialization, see
    # https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa
    if W is None:
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6./(n_in + n_out)),
                high=numpy.sqrt(6./(n_in + n_out)),
                size=(n_in, n_out)
                ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)

    # init biases to positive values, so we should be initially in the linear regime of the linear rectified function
    if b is None:
        b_values = numpy.ones((fshape,), dtype=theano.config.floatX) * numpy.cast[theano.config.floatX](0.01)
        b = theano.shared(value=b_values, name='b', borrow=True)
    return W, b

rng = numpy.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

def rescale_weights(params, incoming_max):
    incoming_max = numpy.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * numpy.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)

def up_sample(overlaps,data_y,step_size):
    data_yy=[]
    for index in range(len(overlaps)):
        value= sum(numpy.asarray(data_y)[overlaps[index]]) / (step_size * len(overlaps[index]))
        data_yy.append(value)
    return numpy.asarray(data_yy)

def convert_to_grayscale(params):
    for dir in params["dataset"]:
        if dir ==-1:
            continue
        im_type='rgb'
        im_type_to='gray'
        new_dir=dir[0]+im_type_to+"/"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            full_path=dir[0]+'/'+im_type+'/*.png'
            lst=glob.glob(full_path)
            for f in lst:
                img = Image.open(f).convert('L')
                img.save(new_dir+os.path.basename(f))
            print("data set converted %s"%(dir[0]))
        else:
            print("data set has already converted %s"%(dir[0]))
    print "Gray scale conversation completed"


def start_log(datasets,params):
    log_file=params["log_file"]
    create_file(log_file)

    X_train, y_train,overlaps_train = datasets[0]
    X_val, y_val,overlaps_val = datasets[1]
    X_test, y_test,overlaps_test = datasets[2]
    y_train_mean=np.mean(y_train)
    y_train_abs_mean=np.mean(np.abs(y_train))
    y_val_mean=np.mean(y_val)
    y_val_abs_mean=np.mean(np.abs(y_val))
    y_test_mean=np.mean(y_test)
    y_test_abs_mean=np.mean(np.abs(y_test))
    ds= get_time()

    log_write("Run Id: %s"%(params['rn_id']),params)
    log_write("Deployment notes: %s"%(params['notes']),params)
    log_write("Running mode: %s"%(params['run_mode']),params)
    log_write("Running model: %s"%(params['model']),params)
    log_write("Image type: %s"%(params["im_type"]),params)
    log_write("Batch size: %s"%(params['batch_size']),params)
    log_write("Images are cropped to: %s"%(params['size']),params)
    log_write("Dataset splits: %s"%(params["step_size"]),params)
    log_write("List of dataset used:",params)

    for dir in params["dataset"]:
        if dir==-1:
            continue
        log_write(dir[0],params)

    log_write("Starting Time:%s"%(ds),params)
    log_write("size of training data:%f"%(len(X_train)),params)
    log_write("size of val data:%f"%(len(X_val)),params)
    log_write("size of test data:%f"%(len(X_test)),params)
    log_write("Mean of training data:%f, abs mean: %f"%(y_train_mean,y_train_abs_mean),params)
    log_write("Mean of val data:%f, abs mean: %f"%(y_val_mean,y_val_abs_mean),params)
    log_write("Mean of test data:%f, abs mean: %f"%(y_test_mean,y_test_abs_mean),params)

def get_time():
    return str(datetime.datetime.now().time()).replace(":","-").replace(".","-")

def get_map_loc():
    ind=randint(0,512)
    return ind
def get_patch_loc(params):
    patch_margin=params["patch_margin"]
    orijinal_size=params['orijinal_size']
    size=params['size']
    x1=randint(patch_margin[0],orijinal_size[0]-(patch_margin[0]+size[0]))
    x2=x1+size[0]

    y1=randint(patch_margin[1],orijinal_size[1]-(patch_margin[1]+size[1]))
    y2=y1+size[1]
    return (x1,y1,x2,y2)

def create_file(log_file):
    log_dir= os.path.dirname(log_file)
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if(os.path.isfile(log_file)):
        with open(log_file, "w"):
            pass
    else:
        os.mknod(log_file)

def log_to_file(str,params):
    with open(params["log_file"], "a") as log:
        log.write(str)

def log_write(str,params):
    print(str)
    ds= get_time()
    str=ds+" | "+str+"\n"
    log_to_file(str,params)


def log_read(mode,params):
    wd=params["wd"]
    filename=params['log_file']
    with open(wd+"/logs/"+filename) as file:
        data = file.read()
        lines = data.split("\n")
        i=0
        list=[]
        for line in lines:
            if mode+"-->" in line:
                epoch=0
                error=0.
                sl=line.split("|")
                for s in sl:
                    if "epoch" in s:
                        epoch=int(s.strip().split(" ")[2])
                    if "error" in s:
                        error=float(s.strip().split(" ")[1])
                list.append((epoch,error))
    #numpy.array([[epoch,error] for (epoch,error) in list_val])[:,1] #all error
    return list

def log_read_train(params):
    wd=params["wd"]
    mode="TRAIN"
    filename=params['log_file']
    with open(wd+"/logs/"+filename) as file:
        data = file.read()
        lines = data.split("\n")
        i=0

        list=[]
        for line in lines:
            if mode+"-->" in line:
                epoch=0
                batch_index=0
                error=0.
                sl=line.split("|")
                for s in sl:
                    if "epoch" in s:
                        epoch=int(s.strip().split(" ")[2])
                    if "error" in s:
                        error=float(s.strip().split(" ")[1])
                    if "minibatch" in s:
                        batch_index=int(s.strip().split(" ")[1].split("/")[0])
                list.append((epoch,batch_index,error))
    #numpy.array([[b, c, d] for (b, c, d) in list_val if b==1 ])[:,2] #first epoch all error
    return list


def huber(y_true, y_pred, epsilon=0.1):
    """Huber regression loss
    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).
    http://en.wikipedia.org/wiki/Huber_Loss_Function
    """
    abs_r = T.abs_(y_pred - y_true)
    loss = 0.5 * abs_r ** 2
    idx = abs_r <= epsilon
    loss[idx] = epsilon * abs_r[idx] - 0.5 * epsilon ** 2
    return loss

def alpha_huber(y_true, y_pred):
    """ sets the epislon in huber loss equal to a percentile of the residuals
    """
    # abs_r = T.abs_(y_pred - y_true)
    # loss = 0.5 * T.sqr(abs_r)
    # epsilon = np.percentile(loss, alpha * 100)
    # idx = abs_r <= epsilon
    # loss[idx] = epsilon * abs_r[idx] - 0.5 * T.sqr(epsilon)
    #switch(cond, ift, iff)
    alpha=0.95
    abs_r = T.abs_(y_pred - y_true)
    epsilon = np.percentile(0.5 * T.sqr(abs_r), alpha * 100)
    loss =T.switch(T.le(abs_r,epsilon),epsilon * abs_r - 0.5 * T.sqr(epsilon),0.5 * T.sqr(abs_r))

    return loss

def epsilon_insensitive(y_true,y_pred, epsilon):
    """Epsilon-Insensitive loss (used by SVR).
    loss = max(0, |y - p| - epsilon)
    """
    loss = T.maximum(T.abs_(y_true-y_pred)-epsilon,0)
    return loss


def mean_squared_epislon_insensitive(y_true, y_pred):
    """Epsilon-Insensitive loss.
    loss = max(0, |y - p| - epsilon)^2
    """
    epsilon=0.05
    return T.mean(T.sqr(epsilon_insensitive(y_true, y_pred, epsilon)))

msei=mean_squared_epislon_insensitive
ah=alpha_huber
