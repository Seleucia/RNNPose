from model.lstm import  lstm
from model.lstmnp import  lstmnp
from model.lstm2 import lstm2
from model.erd import erd
from model.gru import gru
from model.erd_pre import erd_pre
from helper.optimizer import ClipRMSprop, RMSprop
import helper.utils as u
import theano

def get_model(params):
    if(params["model"]=="lstm"):
        model = lstm(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse',optimizer=RMSprop)
    elif(params["model"]=="lstmnp"):
        model = lstmnp(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse',optimizer=RMSprop)
    elif(params["model"]=="lstm2"):
        model = lstm2(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
    elif(params["model"]=="erd"):
        model = erd(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
    elif(params["model"]=="erd_pre"):
        model = erd_pre(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
    elif(params["model"]=="gru"):
        model = gru(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse',optimizer=RMSprop)
    else:
        model=None
    return model

def get_model_pretrained(params):
    mparams=u.read_params(params)
    model=get_model(params)
    model=u.set_params(model,mparams)
    return model
