import helper.utils as utils
import helper.config as config
import plot_data

params=config.get_params()
params["log_file"]="cnn_lstm_resume_0_18-12-44-121954.txt"

lim=300
model="VAL"
list_val=utils.log_read(model,params)
plot_data.plot_val(list_val,params["wd"]+"/"+"logs/img/"+params["log_file"].replace(".txt",".png"),lim=lim)

list_val=utils.log_read_train(params)
plot_data.plot_val(list_val,params["wd"]+"/"+"logs/img/"+params["log_file"].replace(".txt",".png"),tt='Training L2 loss',lim=lim)
