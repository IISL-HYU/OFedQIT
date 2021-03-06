from numpy import gradient
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

from .quantize import quantize_gradient_sum

class CustomModelList_Regression(list):
  def Lpfed_avg(self, x, y, central_server, prob, L, marker):
    trainable_vars = central_server.model.trainable_variables
    loss_avg = 0
    for i, model in enumerate(self):
      train_results = model.train_step(x[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))],y[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))])
      model.loss_temp_list.append(train_results[1])
      loss_temp = np.mean(model.loss_temp_list)
      loss_avg += loss_temp
      # Averaging gradients
      if marker % L == 0:
        model.gradient_sum = train_results[0]
      else:
        tmp = [x + y for x, y in zip(model.gradient_sum, train_results[0])]
        model.gradient_sum = tmp
    loss_avg = loss_avg / len(self)
    ## Update weights 
    if (marker + 1) % L == 0:
      random_list = randomize_list(len(self), prob)
      randomized_models = []
      gradient_avg = 0
      for i, model in enumerate(self):
        if(random_list[i] != 0):
          randomized_models.append(self[i].model)
          if(gradient_avg == 0):
            gradient_avg = model.gradient_sum
          else:
            for j in range(len(model.gradient_sum)):
              gradient_avg[j] = gradient_avg[j] + model.gradient_sum[j]
      if(len(randomized_models) != 0):
        for i in range(len(gradient_avg)):
          gradient_avg[i] = gradient_avg[i] / (len(randomized_models))
        central_server.optimizer.apply_gradients(zip(gradient_avg, trainable_vars))
        for i, model in enumerate(self):
          self[i].model.set_weights(central_server.model.get_weights())
    return loss_avg
        


  # Quantization method included
  def Lpqfed_avg(self, x, y, central_server, prob, L, marker, b):
    trainable_vars = central_server.model.trainable_variables
    loss_avg = 0
    for i, model in enumerate(self):
      train_results = model.train_step(x[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))],y[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))])
      model.loss_temp_list.append(train_results[1])
      loss_temp = np.mean(model.loss_temp_list)
      loss_avg += loss_temp
      # Averaging gradients
      if marker % L == 0:
        model.gradient_sum = train_results[0]
      else:
        tmp = [x + y for x, y in zip(model.gradient_sum, train_results[0])]
        model.gradient_sum = tmp
          
    loss_avg = loss_avg / len(self)
    ## Update weights 
    if (marker + 1) % L == 0:
      random_list = randomize_list(len(self), prob)
      randomized_models = []
      gradient_avg = 0
      for i, model in enumerate(self):
        if(random_list[i] != 0):
          randomized_models.append(self[i].model)
          # Quantize gradient_sum (Algorithm 2 in OFedQIT)
          s = 1
          model.gradient_sum = quantize_gradient_sum(model.gradient_sum, s, b, prob) #(vector, quantization_level, division parameter, probability)
          if(gradient_avg == 0):
            gradient_avg = model.gradient_sum
          else:
            for j in range(len(model.gradient_sum)):
              gradient_avg[j] = gradient_avg[j] + model.gradient_sum[j]
      if(len(randomized_models) != 0):
        for i in range(len(gradient_avg)):
          gradient_avg[i] = gradient_avg[i] / (len(self))
          # pdb.set_trace()
        central_server.optimizer.apply_gradients(zip(gradient_avg, trainable_vars))
        for i, model in enumerate(self):
          self[i].model.set_weights(central_server.model.get_weights())
    return loss_avg


class CustomModel_Regression(keras.Model):
  def __init__(self, model):
      super(CustomModel_Regression, self).__init__()
      self.model = model
      self.gradient_sum = 0
      self.loss_temp_list = [1]
  def train_step(self, x, y):
    loss_fn = keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
      y_pred = self.model(x, training=True)  # Forward pass
      loss = loss_fn(y, y_pred)
    # Compute gradients
    trainable_vars = self.model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    return gradients, loss.numpy()
    # def q_train_step(self, x, y):
    #   loss_fn = keras.losses.MeanSquaredError()
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #   with tf.GradientTape() as tape:
    #     y_pred = self.model(x, training=True)  # Forward pass
    #     # Compute the loss value
    #     # (the loss function is configured in `compile()`)
    #     loss = loss_fn(y, y_pred)
    #   trainable_vars = self.model.trainable_variables
    #   gradients = tape.gradient(loss, trainable_vars)
    #   s, b = 1, 1
    #   grad_len = len(gradients)
    #   q_gradients = [(tf.Variable(gradients[i])) for i in range(grad_len)]
    #   # reshape (1, all_params)
    #   model_params = [None for i in range(len(q_gradients))]
    #   for i in range(len(q_gradients)):
    #     model_params[i] = q_gradients[i].numpy().shape
    #   all_params = []
    #   for i in range(len(q_gradients)):
    #     all_params = np.append(all_params, q_gradients[i])
    #   div_len = math.ceil(len(all_params) / b)  
    #   for i in range(b):
    #     temp_params = all_params[i*div_len:(i+1)*div_len]
    #     temp_params = quantize(temp_params, s)
    #     all_params[i*div_len:(i+1)*div_len] = temp_params
    #   q_gradients_list = [None for i in range(len(model_params))]
    #   bound_bef, bound_aft = 0, 0
    #   for i in range(len(model_params)):
    #     mulp = 1
    #     for j in range(len(model_params[i])):
    #       mulp = mulp * model_params[i][j]
    #     bound_bef = bound_aft
    #     bound_aft = bound_aft + mulp
    #     q_gradients_list[i] = all_params[bound_bef:bound_aft].reshape(model_params[i])
    #   for i in range(grad_len):
    #     # Time check
    #     # work_start = int(time.time() * 1000.0)
    #     q_gradients[i].assign(q_gradients_list[i])
    #   # # Return a dict mapping metric names to current value
    #   return q_gradients, loss.numpy()
  

def randomize_list(n, p):
  select_list = [0, 1]
  distri = [1-p, p]
  random_list = []
  for i in range(n):
    random_list.append(random.choices(select_list, distri)[0])
  return random_list