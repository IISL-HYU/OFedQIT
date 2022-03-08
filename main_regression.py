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
L = 1
L1 = 2
L2 = 2

prob = 1.0
prob2 = 0.5
prob3 = 0.1

# sca_metric = keras.metrics.MeanSquaredError(name="sca")
# p_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="p_sca")
# rp_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="rp_sca")

# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# x_train = x_train.reshape((60000, 28, 28, 1))
# x_test = x_test.reshape((10000, 28, 28, 1))
# x_train, x_test = x_train / 255.0, x_test / 255.0
# dataset = scipy.io.loadmat('./Data/TomData.mat')
# dataset = scipy.io.loadmat('./Data/TwitterDataL.mat')
# dataset = scipy.io.loadmat('./Data/AirData.mat')
# dataset = scipy.io.loadmat('./Data/Wave.mat')
dataset = scipy.io.loadmat('./Data/Conductivity.mat')
X = dataset['X']
y = dataset['y']
X = X.tolist()
y_train = np.array(y).reshape(len(y))
T = len(y)
input_size = len(X)
X_t_train = [list(i) for i in zip(*X)]
X_t_train = np.array(X_t_train).reshape(T, input_size, 1)

reuse = 1
data = "Conductivity"
b = 1

# models with prob=1.0
all_models, central_server = mg.model_generation_regression(N, input_size)
# q_all_models, q_central_server = mg.model_generation_regression(N, input_size)
q2_all_models, q2_central_server = mg.model_generation_regression(N, input_size)

# models with prob=0.5
# p2_all_models, p2_central_server = mg.model_generation_regression(N, input_size)
# q_p2_all_models, q_p2_central_server = mg.model_generation_regression(N, input_size)
# q2_p2_all_models, q2_p2_central_server = mg.model_generation_regression(N, input_size)

# models with prob=0.1
p3_all_models, p3_central_server = mg.model_generation_regression(N, input_size)
# q_p3_all_models, q_p3_central_server = mg.model_generation_regression(N, input_size)
q2_p3_all_models, q2_p3_central_server = mg.model_generation_regression(N, input_size)

loss_list = [1]
q_loss_list = [1]
q2_loss_list = [1]

p2_loss_list = [1]
q_p2_loss_list = [1]
q2_p2_loss_list = [1]

p3_loss_list = [1]
q_p3_loss_list = [1]
q2_p3_loss_list = [1]

T = math.floor(T/N)

      # for u in range(Nuser):
      #   theta[u], error_sel[u][t], kernel_loss[u], select_index[u] = MKOFL_Function_sing(y[(t)*Nuser+u], X_t[(t)*Nuser+u], params, theta_int[u], kernel_loss_int[u], D, global_prob)

      #   if t==0:
      #     error_sel[u][t] = 1 # Initial MSE values

      #   ofl_sel[u][t] = np.mean(error_sel[u][0:t+1])
      #   kernel_loss_int[u] = kernel_loss[u]
      #   theta_int[u] = theta[u]

for iter in range(reuse):
  for i in range(T):
    x = X_t_train[N*(i):N*(i+1)]
    y = y_train[N*(i):N*(i+1)]

    # Benchmark model (OFedAvg)
    results = all_models.Lpfed_avg(x, y, central_server, prob, L, i)
    loss_list.append(results)
    # OFedQIT model I (L=1)
    # q_results = q_all_models.Lpqfed_avg(x, y, q_central_server, prob, L1, i, b)
    # q_loss_list.append(q_results)
    # OFedQIT model II (L=10)
    q2_results = q2_all_models.Lpqfed_avg(x, y, q2_central_server, prob, L2, i, b)
    q2_loss_list.append(q2_results)
    
    # # Benchmarking model(p=0.5)
    # p2_results = p2_all_models.Lpfed_avg(x, y, p2_central_server, prob2, L, i)
    # p2_loss_list.append(p2_results)
    # # OFedQIT model I (L=1)
    # q_p2_results = q_p2_all_models.Lpqfed_avg(x, y , q_p2_central_server, prob2, L1, i, b)
    # q_p2_loss_list.append(q_p2_results)
    # # # OFedQIT model II (L=3)
    # q2_p2_results = q2_p2_all_models.Lpqfed_avg(x, y, q2_p2_central_server, prob2, L2, i, b)
    # q2_p2_loss_list.append(q2_p2_results)
    
    # Benchmarking model(p=0.1)
    p3_results = p3_all_models.Lpfed_avg(x, y, p3_central_server, prob3, L, i)
    p3_loss_list.append(p3_results)
    # OFedQIT model I (L=1)
    # q_p3_results = q_p3_all_models.Lpqfed_avg(x, y , q_p3_central_server, prob3, L1, i, b)
    # q_p3_loss_list.append(q_p3_results)
    # OFedQIT model II (L=3)
    q2_p3_results = q2_p3_all_models.Lpqfed_avg(x, y, q2_p3_central_server, prob3, L2, i, b)
    q2_p3_loss_list.append(q2_p3_results)

    if(i % 20 == 0):
      print("iteration : ", iter, ", i : ", i)
      print("loss : %.7f " %(results))
      # print("[Q]loss : %.7f " %(q_results))
      print("[Q2]loss : %.7f " %(q2_results))
      
      # print("loss : %.7f " %(p2_results))
      # print("[Q]loss : %.7f " %(q_p2_results))
      # print("[Q2]loss : %.7f " %(q2_p2_results))
      
      print("loss : %.7f " %(p3_results))
      # print("[Q]loss : %.7f " %(q_p3_results))
      print("[Q2]loss : %.7f " %(q2_p3_results))
    
    
with open(f"./Regression_mse/OFedAvg_{data}_L{L}_p{prob}.pkl","wb") as f:
    pickle.dump(loss_list, f)
    
# with open(f"./Regression_mse/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob}.pkl","wb") as f:
#     pickle.dump(q_loss_list, f)

with open(f"./Regression_mse/OFedQIT_{data}_L{L2}_s1_b{b}_p{prob}.pkl","wb") as f:
    pickle.dump(q2_loss_list, f)
    
    
# with open(f"./Regression_mse/OFedAvg_{data}_L{L}_p{prob2}.pkl","wb") as f:
#     pickle.dump(p2_loss_list, f)
    
# with open(f"./Regression_mse/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob2}.pkl","wb") as f:
#     pickle.dump(q_p2_loss_list, f)

# with open(f"./Regression_mse/OFedQIT_{data}_L{L2}_s1_b{b}_p{prob2}.pkl","wb") as f:
#     pickle.dump(q2_p2_loss_list, f)
    
    
with open(f"./Regression_mse/OFedAvg_{data}_L{L}_p{prob3}.pkl","wb") as f:
    pickle.dump(p3_loss_list, f)
    
# with open(f"./Regression_mse/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob3}.pkl","wb") as f:
#     pickle.dump(q_p3_loss_list, f)

with open(f"./Regression_mse/OFedQIT_{data}_L{L2}_s1_b{b}_p{prob3}.pkl","wb") as f:
    pickle.dump(q2_p3_loss_list, f)
    