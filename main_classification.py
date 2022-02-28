from tensorflow import keras
from keras import datasets
from numpy import array
from numpy.linalg import norm
import pickle
import pdb

import IISL_FLpkg.model_generator_classification as mg

N = 100
L = 10

prob = 0.1

sca_metric = keras.metrics.SparseCategoricalAccuracy(name="sca")
q_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="q_sca")

all_models, central_server = mg.model_generation(N, sca_metric)
q_all_models, q_central_server = mg.model_generation(N, q_sca_metric)

# Setting dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.reshape((50000, 32, 32, 3))
x_test = x_test.reshape((10000, 32, 32, 3))
x_train, x_test = x_train / 255.0, x_test / 255.0

# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# x_train = x_train.reshape((60000, 28, 28, 1))
# x_test = x_test.reshape((10000, 28, 28, 1))
# x_train, x_test = x_train / 255.0, x_test / 255.0

# Initializing metrics
loss_list = []
q_loss_list = []
accuracy_list = []
q_accuracy_list = []

for iter in range(3):
  for i in range(500):
    x = x_train[100*(i):100*(i+1)]
    y = y_train[100*(i):100*(i+1)]

    # Benchmarking model
    results = all_models.Lpfed_avg(x, y, sca_metric, central_server, prob, 1, i)
    loss_list.append(results[0])
    accuracy_list.append(results[1])
    # OFedQIT model
    q_results = q_all_models.Lpqfed_avg(x, y, q_sca_metric, q_central_server, prob, L, i)
    q_loss_list.append(q_results[0])
    q_accuracy_list.append(q_results[1])

    if(i % 20 == 0):
      print("iteration : ", iter, ", i : ", i)
      print("loss : %.7f, sca : %.7f" %( results[0], results[1]))
      print("[Q]loss : %.7f, sca : %.7f" %( q_results[0], q_results[1]))
    
with open("./Classification_Acc_dohyeok/OFedAvg_CIFAR10_p0.1.pkl","wb") as f:
    pickle.dump(accuracy_list, f)
    
with open("./Classification_Acc_dohyeok/OFedQIT_CIFAR10_L10_s1_p0.1.pkl","wb") as f:
    pickle.dump(q_accuracy_list, f)