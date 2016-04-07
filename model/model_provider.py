from model.lstm import  lstm
from model.lstmnp import  lstmnp
from model.lstm2erd import lstm2erd
from model.lstm2 import lstm2
from model.erd import erd
from model.gru import gru
from model.egd import egd
from model.blstmnp import blstmnp
from model.erd_pre import erd_pre
from model.cnn_lstm import cnn_lstm
from helper.optimizer import ClipRMSprop, RMSprop,Adam
import helper.utils as u
import theano

def get_model(params,rng):
    if(params["model"]=="lstm"):
        model = lstm(rng,n_in=1024, n_lstm= params['n_hidden'],n_out= params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="lstmnp"):
        model = lstmnp(1024, params['n_hidden'], params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="lstm2"):
        model = lstm2(1024, params['n_hidden'], params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="erd"):
        model = erd(1024, params['n_hidden'], params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="erd_pre"):
        model = erd_pre(1024, params['n_hidden'], params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="gru"):
        model = gru(rng=rng,n_in=1024, n_lstm=params['n_hidden'],n_out= params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="egd"):
        model = egd(1024, params['n_hidden'], params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="lstm2erd"):
        model = lstm2erd(1024, params['n_hidden'], params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="blstmnp"):
        model = blstmnp(1024, params['n_hidden'], params['n_output'],batch_size=params['batch_size'],lr=params['lr'], optimizer=Adam)
    elif(params["model"]=="cnn_lstm"):
        model = cnn_lstm(rng=rng,params=params,optimizer=Adam)
    else:
        model=None
    return model

def get_model_pretrained(params,rng):
    mparams=u.read_params(params)
    model=get_model(params,rng)
    model=u.set_params(model,mparams)
    return model
