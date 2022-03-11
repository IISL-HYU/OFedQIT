from tensorflow import keras
from tensorflow.keras import datasets
from numpy import array
from numpy.linalg import norm
import pickle
import pdb
import pandas as pd
import numpy as np
import math
import scipy.io

import IISL_FLpkg.model_generator_classification as mg

N = 100
L = 1
L1 = 1
# L2 = 2

prob = 1.0
# prob2 = 0.5
prob3 = 0.1

# Setting dataset

# Occupancy Estimation dataset
# trainData = pd.read_csv("data/Occupancy_Estimation.csv")
# x_train = trainData.iloc[:,:14]
# x_train = x_train.values
# y_train = trainData.iloc[:,14]
# y_train = y_train.array
# input_size = len(x_train[0])
# data = "Occp_v3"
# reuse = 30
# b = 100
# code = 1
# T = math.floor(len(y_train)/(N))

# # MovementData
# dataset = scipy.io.loadmat('./Data/MovementData.mat')
# X = dataset['X']
# y = dataset['y']
# X = X.tolist()
# y_train = np.array(y).reshape(len(y))
# T = len(y)
# input_size = len(X)
# x_train = [list(i) for i in zip(*X)]
# x_train = np.array(x_train).reshape(T, input_size)
# data = "Move"
# reuse = 1
# b = 1
# code = 3
# T = math.floor(len(y_train)/(N))

# CTG dataset
trainData = pd.read_csv("data/CTG.csv")
x_train = trainData.iloc[:,:22]
x_train = x_train.values
y_train = trainData.iloc[:,22]
y_train = y_train.array
input_size = len(x_train[0])
data = "CTG"
reuse = 30
b = 100
code = 4
T = math.floor(len(y_train)/(N))
# pdb.set_trace()
# CIFAR dataset
# (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# x_train = x_train.reshape((50000, 32, 32, 3))
# x_test = x_test.reshape((10000, 32, 32, 3))
# x_train, x_test = x_train / 255.0, x_test / 255.0

# MNIST dataset
# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# x_train = x_train.reshape((60000, 28, 28, 1))
# x_test = x_test.reshape((10000, 28, 28, 1))
# x_train, x_test = x_train / 255.0, x_test / 255.0
# input_size = len(x_train[0])
# T = math.floor(len(y_train)/N)
# data = "MNIST"
# b = 1000
# code = 2

# models with prob=1.0
all_models, central_server = mg.model_generation(N, input_size, code)
# q_all_models, q_central_server = mg.model_generation(N, input_size, code)
q2_all_models, q2_central_server = mg.model_generation(N, input_size, code)

# # models with prob=0.5
# p2_all_models, p2_central_server = mg.model_generation(N, input_size, code)
# q_p2_all_models, q_p2_central_server = mg.model_generation(N, input_size, code)
# q2_p2_all_models, q2_p2_central_server = mg.model_generation(N, input_size, code)

# # models with prob=0.1
p3_all_models, p3_central_server = mg.model_generation(N, input_size, code)
# q_p3_all_models, q_p3_central_server = mg.model_generation(N, input_size, code)
q2_p3_all_models, q2_p3_central_server = mg.model_generation(N, input_size, code)


# Initializing metrics
loss_list = []
q_loss_list = []
q2_loss_list = []
accuracy_list = []
q_accuracy_list = []
q2_accuracy_list = []

# p2_loss_list = []
# q_p2_loss_list = []
# q2_p2_loss_list = []
# p2_accuracy_list = []
# q_p2_accuracy_list = []
# q2_p2_accuracy_list = []

p3_loss_list = []
q_p3_loss_list = []
q2_p3_loss_list = []
p3_accuracy_list = []
q_p3_accuracy_list = []
q2_p3_accuracy_list = []

for iter in range(reuse):
    for i in range(T):

        x = x_train[N*i:N*(i+1)]
        y = y_train[N*i:N*(i+1)]
        # for j in range(reuse):
        #     if j == 0:
        #         x = x_train[int((N/reuse)*(i)):int((N/reuse)*(i+1))]
        #         y = y_train[int((N/reuse)*(i)):int((N/reuse)*(i+1))]
        #     else :
        #         x = x + x_train[int((N/reuse)*(i)):int((N/reuse)*(i+1))]
        #         y = y + y_train[int((N/reuse)*(i)):int((N/reuse)*(i+1))]
        # x = np.array(x)
        # y = np.array(y)
        # pdb.set_trace()
        # Benchmarking model(p=0.1)
        # results = all_models.Lpfed_avg(x, y, central_server, prob, L, i)
        # loss_list.append(results[0])
        # accuracy_list.append(results[1])
        # # OFedQIT model I (L=1)
        # # q_results = q_all_models.Lpqfed_avg(x, y , q_central_server, prob, L1, i, b)
        # # q_loss_list.append(q_results[0])
        # # q_accuracy_list.append(q_results[1])
        # # OFedQIT model II (L=3)
        # q2_results = q2_all_models.Lpqfed_avg(x, y, q2_central_server, prob, L1, i, b)
        # q2_loss_list.append(q2_results[0])
        # q2_accuracy_list.append(q2_results[1])

        # # Benchmarking model(p=0.5)
        # p2_results = p2_all_models.Lpfed_avg(x, y, p2_central_server, prob2, L, i)
        # p2_loss_list.append(p2_results[0])
        # p2_accuracy_list.append(p2_results[1])
        # # OFedQIT model I (L=1)
        # q_p2_results = q_p2_all_models.Lpqfed_avg(x, y , q_p2_central_server, prob2, L1, i, b)
        # q_p2_loss_list.append(q_p2_results[0])
        # q_p2_accuracy_list.append(q_p2_results[1])
        # # OFedQIT model II (L=3)
        # q2_p2_results = q2_p2_all_models.Lpqfed_avg(x, y, q2_p2_central_server, prob2, L2, i, b)
        # q2_p2_loss_list.append(q2_p2_results[0])
        # q2_p2_accuracy_list.append(q2_p2_results[1])

        # Benchmarking model
        # p3_results = p3_all_models.Lpfed_avg(x, y, p3_central_server, prob3, L, i)
        # p3_loss_list.append(p3_results[0])
        # p3_accuracy_list.append(p3_results[1])
        # # OFedQIT model I (L=1)
        # q_p3_results = q_p3_all_models.Lpqfed_avg(x, y , q_p3_central_server, prob3, L1, i, b)
        # q_p3_loss_list.append(q_p3_results[0])
        # q_p3_accuracy_list.append(q_p3_results[1])
        # OFedQIT model II (L=3)
        q2_p3_results = q2_p3_all_models.Lpqfed_avg(x, y, q2_p3_central_server, prob3, L1, i, b)
        q2_p3_loss_list.append(q2_p3_results[0])
        q2_p3_accuracy_list.append(q2_p3_results[1])

        if(i % 20 == 0):
            print("iter : ", iter, ", i : ", i)
            # print("loss : %.7f, sca : %.7f" %( results[0], results[1]))
            # # print("[Q]loss : %.7f, sca : %.7f" %( q_results[0], q_results[1]))
            # print("[Q2]loss : %.7f, sca : %.7f" %( q2_results[0], q2_results[1]))

            # # print("loss : %.7f, sca : %.7f" %( p2_results[0], p2_results[1]))
            # # print("[Q_P2]loss : %.7f, sca : %.7f" %( q_p2_results[0], q_p2_results[1]))
            # # print("[Q2_P2]loss : %.7f, sca : %.7f" %( q2_p2_results[0], q2_p2_results[1]))

            # print("loss : %.7f, sca : %.7f" %( p3_results[0], p3_results[1]))
            # print("[Q_P3]loss : %.7f, sca : %.7f" %( q_p3_results[0], q_p3_results[1]))
            print("[Q2_P3]loss : %.7f, sca : %.7f" %( q2_p3_results[0], q2_p3_results[1]))

accuracy_list.insert(0, 0)
# q_accuracy_list.insert(0, 0)
q2_accuracy_list.insert(0, 0)

# p2_accuracy_list.insert(0, 0)
# q_p2_accuracy_list.insert(0, 0)
# q2_p2_accuracy_list.insert(0, 0)

p3_accuracy_list.insert(0, 0)
# q_p3_accuracy_list.insert(0, 0)
q2_p3_accuracy_list.insert(0, 0)


# with open(f"./Classification_Acc/OFedAvg_{data}_L{L}_p{prob}.pkl","wb") as f:
#     pickle.dump(accuracy_list, f)

# # # with open(f"./Classification_Acc/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob}.pkl","wb") as f:
# # #     pickle.dump(q_accuracy_list, f)

# with open(f"./Classification_Acc/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob}.pkl","wb") as f:
#     pickle.dump(q2_accuracy_list, f)


# with open(f"./Classification_Acc/OFedAvg_{data}_L{L}_p{prob2}.pkl","wb") as f:
#     pickle.dump(p2_accuracy_list, f)

# with open(f"./Classification_Acc/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob2}.pkl","wb") as f:
#     pickle.dump(q_p2_accuracy_list, f)

# with open(f"./Classification_Acc/OFedQIT_{data}_L{L2}_s1_b{b}_p{prob2}.pkl","wb") as f:
#     pickle.dump(q2_p2_accuracy_list, f)


# with open(f"./Classification_Acc/OFedAvg_{data}_L{L}_p{prob3}.pkl","wb") as f:
#     pickle.dump(p3_accuracy_list, f)

# # with open(f"./Classification_Acc/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob3}.pkl","wb") as f:
# #     pickle.dump(q_p3_accuracy_list, f)

with open(f"./Classification_Acc/OFedQIT_{data}_L{L1}_s1_b{b}_p{prob3}.pkl","wb") as f:
    pickle.dump(q2_p3_accuracy_list, f)