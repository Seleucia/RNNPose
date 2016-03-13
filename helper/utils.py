import numpy
import theano
import theano.tensor as T
import os
import datetime
import numpy as np
from theano import shared
import pickle

dtype = T.config.floatX


def numpy_floatX(data):
    return numpy.asarray(data, dtype=T.config.floatX)

def init_bias(n_out, sample='zero'):
    if sample == 'zero':
        b = np.zeros((n_out,), dtype=dtype)
    elif sample == 'one':
        b = np.ones((n_out,), dtype=dtype)
    elif sample == 'uni':
        b=shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_out)))
    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)
    b = theano.shared(value=b, name='b')
    return b


def init_pweight(shape, name, sample='glorot', seed=None):
    rng = np.random.RandomState(seed)

    if sample == 'glorot':
        ''' Reference: Glorot & Bengio, AISTATS 2010
        '''
        fan_in, fan_out = shape[0],shape[1]
        s = np.sqrt(2. / (fan_in + fan_out))
        values=np.random.normal(loc=0.0, scale=s, size=(shape[0],)).astype(dtype)
    elif sample == 'ortho':
        ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
        '''
        scale=1.1
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        values= np.asarray(scale * q[:shape[0], :shape[1]], dtype=dtype)
    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)

    return shared(values, name=name, borrow=True)

def init_weight(shape, name, sample='glorot', seed=None):
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

    elif sample == 'glorot':
        ''' Reference: Glorot & Bengio, AISTATS 2010
        '''
        fan_in, fan_out = shape[0],shape[1]
        s = np.sqrt(2. / (fan_in + fan_out))
        values=np.random.normal(loc=0.0, scale=s, size=shape).astype(dtype)
    elif sample == 'ortho':
        ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
        '''
        scale=1.1
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        values= np.asarray(scale * q[:shape[0], :shape[1]], dtype=dtype)

    elif sample == 'ortho1':
        W = rng.uniform(low=-1., high=1., size=shape).astype(dtype)
        u, s, v = numpy.linalg.svd(W)
        values= u.astype(dtype)

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

def get_err_fn(self,cost_function,Y):
       cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
       nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))

       tmp = self.output - Y
       tmp = theano.tensor.switch(theano.tensor.isnan(tmp),0,tmp)
       mse = T.mean((tmp) ** 2)

       cost = 0
       if cost_function == 'mse':
           cost = mse
       elif cost_function == 'cxe':
           cost = cxe
       else:
           cost = nll
       return cost

def get_loss(gt,est):
    batch_size=gt.shape[0]
    seq_length=gt.shape[1]
    loss=0
    for b in range(batch_size):
        for s in range(seq_length):
            diff_vec=np.abs(gt[b][s].reshape(13,3) - est[b][s].reshape(13,3)) #13*3
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss +=np.nanmean(sq_m)
    loss/=(seq_length*batch_size)
    return (loss)

def get_loss_bb(gt,est,bb_list):
    batch_size=gt.shape[0]
    seq_length=gt.shape[1]
    loss=0
    bb_loss=0
    counter=0
    loss_list=[]
    for b in range(batch_size):
        for s in range(seq_length):
            diff_vec=np.abs(gt[b][s].reshape(13,3) - est[b][s].reshape(13,3)) #54*3
            bb_vec=bb_list[counter]
            loss_vec=(diff_vec*bb_vec)
            b_l=np.mean(np.sqrt(np.sum(loss_vec**2,axis=1)))
            loss_list.append(b_l)
            bb_loss +=b_l
            loss +=np.mean(np.sqrt(np.sum(diff_vec**2,axis=1)))
    bb_loss/=(seq_length*batch_size)
    loss/=(seq_length*batch_size)

    return (loss,bb_loss,loss_list)

rng = numpy.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))


def rescale_weights(params, incoming_max):
    incoming_max = numpy.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * numpy.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)

def start_log(params):
    log_file=params["log_file"]
    create_file(log_file)

    ds= get_time()

    log_write("Run Id: %s"%(params['rn_id']),params)
    log_write("Deployment notes: %s"%(params['notes']),params)
    log_write("Running mode: %s"%(params['run_mode']),params)
    log_write("Running model: %s"%(params['model']),params)
    log_write("Batch size: %s"%(params['batch_size']),params)
    log_write("Sequence size: %s"%(params['seq_length']),params)

    log_write("Starting Time:%s"%(ds),params)
    log_write("size of training data:%f"%(params["len_train"]),params)
    log_write("size of test data:%f"%(params["len_test"]),params)

def get_time():
    return str(datetime.datetime.now().time()).replace(":","-").replace(".","-")

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

def write_params(mparams,params,ext):
    wd=params["wd"]
    filename=params['model']+"_"+params["rn_id"]+"_"+ext
    fpath=wd+"/cp/"+filename
    if os.path.exists(fpath):
        os.remove(fpath)
    with open(fpath,"a") as f:
        pickle.dump([param.get_value() for param in mparams], f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved"+filename)


def read_params(params):
    wd=params["wd"]
    with open(wd+"/cp/"+params['mfile']) as f:
        mparams=pickle.load(f)
        return mparams

def set_params(model,mparams):
    counter=0
    for p in mparams:
        if(counter<(len(mparams)-2)):
                model.params[counter].set_value(p)
        # if(counter==(len(mparams)-1)):
        #     model.params[counter].set_value(p[0:39])
        # elif(counter==(len(mparams)-2)):
        #     model.params[counter].set_value(p[:,0:39])
        # else:
        #     model.params[counter].set_value(p)
        counter=counter+1
    return model




