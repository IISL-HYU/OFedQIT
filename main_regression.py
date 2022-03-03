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
L1 = 1
L2 = 10

prob = 0.1
# sca_metric = keras.metrics.MeanSquaredError(name="sca")
# p_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="p_sca")
# rp_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="rp_sca")


# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# x_train = x_train.reshape((60000, 28, 28, 1))
# x_test = x_test.reshape((10000, 28, 28, 1))
# x_train, x_test = x_train / 255.0, x_test / 255.0
# dataset = scipy.io.loadmat('./Data/TomData.mat')
# dataset = scipy.io.loadmat('./Data/TwitterDataL.mat')
dataset = scipy.io.loadmat('./Data/AirData.mat')
# dataset = scipy.io.loadmat('./Data/Wave.mat')
# dataset = scipy.io.loadmat('./Data/Conductivity.mat')
X = dataset['X']
y = dataset['y']
X = X.tolist()
y_train = np.array(y).reshape(len(y))
T = len(y)
input_size = len(X)
X_t_train = [list(i) for i in zip(*X)]
X_t_train = np.array(X_t_train).reshape(T, input_size, 1)
all_models, central_server = mg.model_generation_regression(N, input_size)
q_all_models, q_central_server = mg.model_generation_regression(N, input_size)
q2_all_models, q2_central_server = mg.model_generation_regression(N, input_size)
# rp_all_models, rp_central_server = mg.model_generation(N, rp_sca_metric)

loss_list = [1]
q_loss_list = [1]
q2_loss_list = [1]

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

    # Benchmark model (OFedAvg)
    results = all_models.Lpfed_avg(x, y, central_server, prob, 1, i)
    loss_list.append(results)
    # OFedQIT model I (L=1)
    q_results = q_all_models.Lpqfed_avg(x, y, q_central_server, prob, L1, i)
    q_loss_list.append(q_results)
    # OFedQIT model II (L=10)
    q2_results = q2_all_models.Lpqfed_avg(x, y, q2_central_server, prob, L2, i)
    q2_loss_list.append(q2_results)

    if(i % 10 == 0):
      print("iteration : ", iter, ", i : ", i)
      print("loss : %.7f " %(results))
      print("[Q]loss : %.7f " %( q_results))
      print("[Q2]loss : %.7f " %( q2_results))
    
with open("./Regression_mse/OFedAvg_Air_p0.1.pkl","wb") as f:
    pickle.dump(loss_list, f)
    
with open("./Regression_mse/OFedQIT_Air_L1_s1_p0.1.pkl","wb") as f:
    pickle.dump(q_loss_list, f)

with open("./Regression_mse/OFedQIT_Air_L10_s1_p0.1.pkl","wb") as f:
    pickle.dump(q2_loss_list, f)
    