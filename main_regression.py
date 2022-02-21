from tensorflow import keras
from keras import datasets
#from tensorflow.python.keras impordatasets
import numpy as np
from numpy import array
from numpy.linalg import norm
import pickle
import scipy.io
import IISL_FLpkg.model_generator_regression as mg
import pdb
import math

N = 100
sca_metric = keras.metrics.MeanSquaredError(name="sca")
# p_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="p_sca")
# rp_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="rp_sca")


# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# x_train = x_train.reshape((60000, 28, 28, 1))
# x_test = x_test.reshape((10000, 28, 28, 1))
# x_train, x_test = x_train / 255.0, x_test / 255.0
dataset = scipy.io.loadmat('./Data/Conductivity.mat')
# dataset = scipy.io.loadmat('./Data/Wave.mat')
X = dataset['X']
y = dataset['y']
X = X.tolist()
y_train = np.array(y).reshape(len(y))
T = len(y)
input_size = len(X)
X_t_train = [list(i) for i in zip(*X)]
X_t_train = np.array(X_t_train).reshape(T, input_size, 1)
all_models, central_server = mg.model_generation_regression(N, input_size)



# p_all_models, p_central_server = mg.model_generation(N, p_sca_metric)
# rp_all_models, rp_central_server = mg.model_generation(N, rp_sca_metric)

loss_list = [1]
# loss_temp_list = [1]
p_loss_list = []
rp_loss_list = []

# metric_loss_list = [1]
# p_accuracy_list = []
# rp_accuracy_list = []

T = math.floor(T/N)

      # for u in range(Nuser):
      #   theta[u], error_sel[u][t], kernel_loss[u], select_index[u] = MKOFL_Function_sing(y[(t)*Nuser+u], X_t[(t)*Nuser+u], params, theta_int[u], kernel_loss_int[u], D, global_prob)

      #   if t==0:
      #     error_sel[u][t] = 1 # Initial MSE values

      #   ofl_sel[u][t] = np.mean(error_sel[u][0:t+1])
      #   kernel_loss_int[u] = kernel_loss[u]
      #   theta_int[u] = theta[u]

for iter in range(1):
  for i in range(T):
    x = X_t_train[N*(i):N*(i+1)]
    y = y_train[N*(i):N*(i+1)]

    # pdb.set_trace()
    results = all_models.fed_avg(x, y, central_server)
    # loss_temp_list.append(results)
    # loss_temp = np.mean(loss_temp_list[0:i+2])
    loss_list.append(results)
    # metric_loss_list.append(results[1])
    # p_results = p_all_models.pfed_avg(x, y, p_sca_metric, p_central_server, L, i)
    # p_loss_list.append(p_results[0])
    # p_accuracy_list.append(p_results[1])

    # rp_results = rp_all_models.rpfed_avg(x, y, rp_sca_metric, rp_central_server, p, L, i)
    # rp_loss_list.append(rp_results[0])
    # rp_accuracy_list.append(rp_results[1])
    
    # r_results = r_all_models.rfed_avg(x, y, r_central_server, 0.1)
    # r_loss_list.append(r_results[0])
    # r_accuracy_list.append(r_results[1])

    # rq_results = rq_all_models.rqfed_avg(x, y, rq_central_server, 0.1)
    # rq_loss_list.append(rq_results[0])
    # rq_accuracy_list.append(rq_results[1])

    if(i % 30 == 0):
      print("iteration : ", iter, ", i : ", i)
      print("loss : %.7f " %( results))
      # print("[P]loss : %.7f, sca : %.7f" %( p_results[0], p_results[1]))
      # print("[RP]loss : %.7f, sca : %.7f" %( rp_results[0], rp_results[1]))
    #   print("[R]loss : %.7f, sca : %.7f" %( r_results[0], r_results[1]))
    #   print("[RQ]loss : %.7f, sca : %.7f" %( rq_results[0], rq_results[1]))
    
# with open("./Regression_mse/OFedAvg_Wave.pkl","wb") as f:
#     pickle.dump(loss_list, f)
    
with open("./Regression_mse/OFedAvg_Conductivity.pkl","wb") as f:
    pickle.dump(loss_list, f)
    
# with open("./Accuracy_lists/OFedIT_rp_sum_centWeight_L.pkl","wb") as f:
#     pickle.dump(rp_accuracy_list, f)