from numpy import gradient
import tensorflow as tf
from tensorflow import keras
import random

from .quantize import quantize_gradient_sum

class CustomModelList_Classification(list):
  def Lpfed_avg(self, x, y, metric, central_server, prob, L, marker):
    trainable_vars = central_server.model.trainable_variables
    loss_avg = 0
    sca_metric_avg = 0
    for i, model in enumerate(self):
      train_results = model.train_step(x[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))],y[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))], metric)
      loss_avg += train_results[1]
      sca_metric_avg += train_results[2]
      # Averaging gradients
      if marker % L == 0:
        model.gradient_sum = train_results[0]
      else:
        tmp = [x + y for x, y in zip(model.gradient_sum, train_results[0])]
        model.gradient_sum = tmp
    loss_avg = loss_avg / len(self)
    sca_metric_avg = sca_metric_avg / len(self)
    
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
    return loss_avg, sca_metric_avg

  def Lpqfed_avg(self, x, y, metric, central_server, prob, L, marker):
    trainable_vars = central_server.model.trainable_variables
    loss_avg = 0
    sca_metric_avg = 0
    for i, model in enumerate(self):
      train_results = model.train_step(x[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))],y[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))], metric)
      loss_avg += train_results[1]
      sca_metric_avg += train_results[2]
      # Averaging gradients
      if marker % L == 0 :
        model.gradient_sum = train_results[0]
      else :
        tmp = [x + y for x, y in zip(model.gradient_sum, train_results[0])]
        model.gradient_sum = tmp
    loss_avg = loss_avg / len(self)
    sca_metric_avg = sca_metric_avg / len(self)
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
          b = 1
          model.gradient_sum = quantize_gradient_sum(model.gradient_sum, s, b, prob) #(vector, quantization_level, division parameter, probability)
          if(gradient_avg == 0):
            gradient_avg = model.gradient_sum
          else:
            for j in range(len(model.gradient_sum)):
              gradient_avg[j] = gradient_avg[j] + model.gradient_sum[j]
      if(len(randomized_models) != 0):
        for i in range(len(gradient_avg)):
          gradient_avg[i] = gradient_avg[i] / (len(self))
          central_server.optimizer.apply_gradients(zip(gradient_avg, trainable_vars))
        for i, model in enumerate(self):
          self[i].model.set_weights(central_server.model.get_weights())
      # # Update metrics (includes the metric that tracks the loss)
      # # Return a dict mapping metric names to current value
    return loss_avg, sca_metric_avg

class CustomModel_Classification(keras.Model):
  def __init__(self, model):
      super(CustomModel_Classification, self).__init__()
      self.model = model
      self.gradient_sum = 0
    
  def train_step(self, x, y, metric):
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
      y_pred = self.model(x, training=True)  # Forward pass
      loss = loss_fn(y, y_pred)
    # Compute gradients
    trainable_vars = self.model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # # Update weights
    result_metric = 0
    metric.update_state(y, y_pred)
    result_metric = metric.result().numpy()
    # # Return a dict mapping metric names to current value
    return gradients, loss.numpy(), result_metric
    

def randomize_list(n, p):
  select_list = [0, 1]
  distri = [1-p, p]
  random_list = []
  for i in range(n):
    random_list.append(random.choices(select_list, distri)[0])
  return random_list