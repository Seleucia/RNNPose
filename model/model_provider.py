from model.lstm import  lstm
import theano

def get_model(params):
    if(params["model"]=="lstm"):
        model = lstm(1024, params['n_hidden'], 54,batch_size=params['batch_size'], single_output=False,output_activation=theano.tensor.nnet.sigmoid, cost_function='mse')
    return model

def get_model_pretrained(params):
    model=get_model(params)
    model.build()
    model_name=params['model_file']+params['model_name']
    model.load_weights(model_name)
    return model
