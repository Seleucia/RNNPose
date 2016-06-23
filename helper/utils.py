import numpy
from random import uniform
import theano
import theano.tensor as T
import os
import datetime
import numpy as np
from theano import shared
import pickle

dtype = T.config.floatX

def rescale_weights(params, incoming_max):
    incoming_max = np.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * np.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=T.config.floatX)

def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'full', 'valid','same'}
    if border_mode == 'full':
        output_length = input_length++ filter_size-1
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    elif border_mode == 'same':
        output_length = input_length
    return (output_length + stride - 1) // stride

      # 'valid'only apply filter to complete patches of the image. Generates
      #  output of shape: image_shape - filter_shape + 1.
      #  'full' zero-pads image to multiple of filter shape to generate output
      #  of shape: image_shape + filter_shape - 1.

def do_nothing(x):
    return x

def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        fan_in = np.prod(shape[1:])
        fan_out = shape[0]
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def init_bias(n_out,rng, sample='zero',name='b'):
    if sample == 'zero':
        b = np.zeros((n_out,), dtype=dtype)
    elif sample == 'one':
        b = np.ones((n_out,), dtype=dtype)
    elif sample == 'uni':
        b=shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_out)))
    elif sample == 'big_uni':
        b=np.asarray(rng.uniform(low=-5,high=5,size=n_out),dtype=dtype);
    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)
    b = theano.shared(value=b, name=name)
    return b

def init_weight(shape, rng,name, sample='glorot', seed=None):
    if sample == 'unishape':
        fan_in, fan_out =get_fans(shape)
        values = rng.uniform(
            low=-np.sqrt(6. / (fan_in + fan_out)),
            high=np.sqrt(6. / (fan_in + fan_out)),
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
        fan_in, fan_out =get_fans(shape)
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

def get_err_fn(self,cost_function,Y):
       cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
       nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))

       tmp = (self.output - Y)*2
       tmp = theano.tensor.switch(theano.tensor.isnan(tmp),0,tmp)
       mse = T.sum((tmp) ** 2)

       cost = 0
       if cost_function == 'mse':
           cost = mse
       elif cost_function == 'cxe':
           cost = cxe
       else:
           cost = nll
       return cost

def prep_pred_file(params):
    f_dir=params["wd"]+"/pred/";
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    f_dir=params["wd"]+"/pred/"+params["model"];
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    map( os.unlink, (os.path.join( f_dir,f) for f in os.listdir(f_dir)) )

def write_auto_pred(est,F_list,params):
    f_dir="/mnt/Data1/hc/auto/"
    ist=0
    for b in range(len(est)):
        action=F_list[b].split('/')[-2]
        sb=F_list[b].split('/')[-3]
        if not os.path.exists(f_dir+sb):
                os.makedirs(f_dir+sb)
        if not os.path.exists(f_dir+sb+'/'+action):
                os.makedirs(f_dir+sb+'/'+action)
        vec=est[b]
        vec_str = '\n'.join(['%f' % num for num in vec])
        p_file=f_dir+sb+'/'+action+'/'+os.path.basename(F_list[b])
        if os.path.exists(p_file):
            print p_file
        with open(p_file, "a") as p:
            p.write(vec_str)


def write_pred(est,bindex,G_list,params):
    batch_size=est.shape[0]
    seq_length=est.shape[1]
    s_index=params["batch_size"]*bindex*seq_length
    f_dir=params["wd"]+"/pred/"+params["model"]+"/"
    for b in range(batch_size):
        for s in range(seq_length):
            diff_vec=est[b][s]*2
            vec_str = ' '.join(['%.6f' % num for num in diff_vec])
            p_file=f_dir+os.path.basename(G_list[s_index])
            with open(p_file, "a") as p:
                p.write(vec_str)
            s_index+=1

def get_loss(gt,est):
    batch_size=gt.shape[0]
    loss=[]
    if(len(gt.shape)==2):
        for b in range(batch_size):
            diff_vec=np.abs(gt[b].reshape(14,3) - est[b].reshape(14,3))*2 #13*3
            diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss.append(np.mean(sq_m))
        loss=np.nanmean(loss)
    else:
        seq_length=gt.shape[1]
        for b in range(batch_size):
            for s in range(seq_length):
                diff_vec=np.abs(gt[b][s].reshape(14,3) - est[b][s].reshape(14,3))*2 #13*3
                diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
                sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
                loss.append(np.mean(sq_m))
        loss=np.nanmean(loss)

    return (loss)


def get_loss(params,gt,est):
    gt= np.asarray(gt)
    # print(gt.shape)
    batch_size=gt.shape[0]
    loss=[]
    if(len(gt.shape)==2):
        for b in range(batch_size):
            diff_vec=np.abs(gt[b].reshape(params['n_output']/3,3) - est[b].reshape(params['n_output']/3,3)) #13*3
            diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss.append(np.mean(sq_m))
        loss=np.nanmean(loss)
    else:
        seq_length=gt.shape[1]
        for b in range(batch_size):
            for s in range(seq_length):
                diff_vec=np.abs(gt[b][s].reshape(params['n_output']/3,3) - est[b][s].reshape(params['n_output']/3,3)) #13*3
                diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
                sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
                loss.append(np.mean(sq_m))
        loss=np.nanmean(loss)

    return (loss)

def get_loss_pred(params,gt,est):
    fest="/home/coskun/PycharmProjects/RNNPoseV2/pred/3.6m/estimation.txt"
    fgt="/home/coskun/PycharmProjects/RNNPoseV2/pred/3.6m/ground_truth.txt"
    loss=0
    loss_list=[]
    with open(fest,"a") as f_handle_est,  open(fgt,"a") as f_handle_gt:
        for b in range(len(gt)):
            diff_vec=np.abs(gt[b].reshape(params['n_output']/3,3) - est[b].reshape(params['n_output']/3,3)) #14,3
            for val in est[b]:
                f_handle_est.write("%f "%(val*1000))
            for val in gt[b]:
                f_handle_gt.write("%f "%(val*1000))
            # val=np.sqrt(np.sum(diff_vec**2,axis=1))
            #
            # for i in range(14):
            #     f=val[i]
            #     f_handle.write("%f"%(f))
            #     if(i<13):
            #         f_handle.write(";")
            f_handle_est.write('\n')
            f_handle_gt.write('\n')
            b_l=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss_list.append(b_l)
            loss +=np.nanmean(np.sqrt(np.sum(diff_vec**2,axis=1)))
        loss=np.nanmean(loss)
    return (loss,loss_list)

def get_loss_bb(gt,est):
    sf="/home/coskun/PycharmProjects/RNNPoseV2/pred/cnn_lstm_s/blanket.txt"
    batch_size=gt.shape[0]
    seq_length=gt.shape[1]
    loss=0
    loss_list=[]
    seq_list=[]
    b_seq_list=[]
    with open(sf,"a") as f_handle:
        for b in range(batch_size):
            seq_los=[0]*seq_length
            for s in range(seq_length):
                diff_vec=np.abs(gt[b][s].reshape(14,3) - est[b][s].reshape(14,3))*2 #14,3
                for val in est[b][s]:
                    f_handle.write("%f "%(val))
                # val=np.sqrt(np.sum(diff_vec**2,axis=1))
                #
                # for i in range(14):
                #     f=val[i]
                #     f_handle.write("%f"%(f))
                #     if(i<13):
                #         f_handle.write(";")
                f_handle.write('\n')
                b_l=np.nanmean(np.sqrt(np.sum(diff_vec**2,axis=1)))
                loss_list.append(b_l)
                seq_los[s]=b_l
                loss +=np.nanmean(np.sqrt(np.sum(diff_vec**2,axis=1)))
            b_seq_list.append(seq_los)
        seq_list=np.mean(b_seq_list,axis=0)
        loss/=(seq_length*batch_size)
    return (loss,loss_list,seq_list)

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
                        epoch=int(s.strip().split(" ")[2])*10
                    if "error" in s:
                        error=float(s.strip().split(' ')[1].replace(',',''))
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
                list.append((epoch,error))
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

def count_params(model_params):
    coun_params=np.sum([np.prod(p.shape.eval()) for p in model_params ])
    return coun_params


def write_params(mparams,params,ext):
    wd=params["wd"]
    filename=params['model']+"_"+params["rn_id"]+"_"+ext
    fpath=wd+"/cp/"+filename
    if os.path.exists(fpath):
        os.remove(fpath)
    with open(fpath,"a") as f:
        pickle.dump([param.get_value() for param in mparams], f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved: "+filename)


def read_params(params):
    wd=params["wd"]
    with open(wd+"/cp/"+params['mfile']) as f:
        mparams=pickle.load(f)
        return mparams

def set_params(model,mparams):
    counter=0
    # for p in mparams[0:-2]:
    for p in mparams:
        model.params[counter].set_value(p)
        # if(counter<(len(mparams)-2)):
        #         model.params[counter].set_value(p)
        # if(counter==(len(mparams)-1)):
        #     model.params[counter].set_value(p[0:39])
        # elif(counter==(len(mparams)-2)):
        #     model.params[counter].set_value(p[:,0:39])
        # else:
        #     model.params[counter].set_value(p)
        counter=counter+1
    return model




