from model.lstm import  lstm
from model.lstm2 import lstm2
from model.erd import erd
from model.erd_pre import erd_pre
from helper.optimizer import ClipRMSprop
import helper.utils as u
import theano

def get_model(params):
    if(params["model"]=="lstm"):
        model = lstm(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse',optimizer=ClipRMSprop)
    if(params["model"]=="lstm2"):
        model = lstm2(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
    if(params["model"]=="erd"):
        model = erd(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
    if(params["model"]=="erd_pre"):
        model = erd_pre(1024, params['n_hidden'], 54,batch_size=params['batch_size'],lr=params['initial_learning_rate'],output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
    return model

def get_model_pretrained(params,model_file):
    params=u.read_params(params,model_file)
    model=get_model(params)
    model.params=u.read_params(params,model_file)
    return model
