from numpy import gradient
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import pdb
from .quantize import quantize

class CustomModelList_Regression(list):
    def fed_avg(self, x, y, central_server):
      trainable_vars = central_server.model.trainable_variables
      loss_avg = 0

      for i, model in enumerate(self):
        train_results = model.train_step(x[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))],y[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))])
        model.loss_temp_list.append(train_results[1])
        loss_temp = np.mean(model.loss_temp_list)
        loss_avg += loss_temp
        # Averaging gradients
        if(i % len(self) == 0):
          gradient_avg = train_results[0]
        else:
          for j in range(len(train_results[0])):
            gradient_avg[j] += train_results[0][j]
      
      for i, model in enumerate(self):
        print(model.loss_temp_list)
      pdb.set_trace()
      loss_avg = loss_avg / len(self)
      for i in range(len(gradient_avg)):
        gradient_avg[i] = gradient_avg[i] / len(self)
      ## Update weights
      central_server.optimizer.apply_gradients(zip(gradient_avg, trainable_vars))
      for i, model in enumerate(self):
        self[i].model.set_weights(central_server.model.get_weights())
        # model.optimizer.apply_gradients(zip(gradient_avg, self[i].model.trainable_variables))
      # # Update metrics (includes the metric that tracks the loss)
      # # Return a dict mapping metric names to current value
      return loss_avg

    # def pfed_avg(self, x, y, metric, central_server, L, marker):
    #   trainable_vars = central_server.model.trainable_variables
    #   loss_avg = 0
    #   sca_metric_avg = 0
    #   for i, model in enumerate(self):
    #     train_results = model.p_train_step(x[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))],y[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))], metric, central_server)
    #     model.optimizer.apply_gradients(zip(train_results[0], self[i].model.trainable_variables))
    #     loss_avg += train_results[1]
    #     sca_metric_avg += train_results[2]
    #     # Averaging gradients
    #     if marker % L == 0 :
    #       model.gradient_sum = train_results[0]
    #     else :
    #       for j in range(len(train_results[0])):
    #         model.gradient_sum[j] += train_results[0][j]
    #   loss_avg = loss_avg / len(self)
    #   sca_metric_avg = sca_metric_avg / len(self)
    #   ## Update weights
    #   if (marker + 1) % L == 0:
    #     gradient_avg = 0
    #     for i, model in enumerate(self):
    #       if(gradient_avg == 0):
    #         gradient_avg = model.gradient_sum
    #       else:
    #         for j in range(len(model.gradient_sum)):
    #           gradient_avg[j] += model.gradient_sum[j]
    #     for i in range(len(gradient_avg)):
    #       gradient_avg[i] = gradient_avg[i] / (len(self))
    #     central_server.optimizer.apply_gradients(zip(gradient_avg, trainable_vars))
    #     for i, model in enumerate(self):
    #       self[i].model.set_weights(central_server.model.get_weights())
    #   # # Update metrics (includes the metric that tracks the loss)
    #   # # Return a dict mapping metric names to current value
    #   return loss_avg, sca_metric_avg

    # def rpfed_avg(self, x, y, metric, rp_central_server, prob, L, marker):
    #   trainable_vars = rp_central_server.model.trainable_variables
    #   loss_avg = 0
    #   sca_metric_avg = 0
    #   for i, model in enumerate(self):
    #     train_results = model.p_train_step(x[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))],y[int(i*len(x)/len(self)):int((i+1)*len(x)/len(self))], metric, rp_central_server)
    #     model.optimizer.apply_gradients(zip(train_results[0], self[i].model.trainable_variables))
    #     loss_avg += train_results[1]
    #     sca_metric_avg += train_results[2]
    #     if marker % L == 0 :
    #       model.gradient_sum = train_results[0]
    #     else :
    #       for j in range(len(train_results[0])):
    #         model.gradient_sum[j] += train_results[0][j]
    #   loss_avg = loss_avg / len(self)
    #   sca_metric_avg = sca_metric_avg / len(self)
    #   ## Update weights
    #   if (marker + 1) % L == 0:
    #     p = prob
    #     random_list = randomize_list(len(self), p)
    #     randomized_models = []
    #     gradient_avg = 0
    #     for i, model in enumerate(self):
    #       if(random_list[i] != 0):
    #         randomized_models.append(self[i].model)
    #         if(gradient_avg == 0):
    #           gradient_avg = model.gradient_sum
    #         else:
    #           for j in range(len(model.gradient_sum)):
    #             gradient_avg[j] += model.gradient_sum[j]
    #     if(len(randomized_models) != 0):
    #       for i in range(len(gradient_avg)):
    #         gradient_avg[i] = gradient_avg[i] / (len(randomized_models))
    #       rp_central_server.optimizer.apply_gradients(zip(gradient_avg, trainable_vars))
    #       for i, model in enumerate(self):
    #         self[i].model.set_weights(rp_central_server.model.get_weights())
    #   return loss_avg, sca_metric_avg


class CustomModel_Regression(keras.Model):
    def __init__(self, model):
        super(CustomModel_Regression, self).__init__()
        self.model = model
        self.gradient_sum = 0
        self.loss_temp_list = [1]
    def train_step(self, x, y):
      loss_fn = keras.losses.MeanSquaredError()
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
      with tf.GradientTape() as tape:
        y_pred = self.model(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = loss_fn(y, y_pred)
      # Compute gradients
      trainable_vars = self.model.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      # # Update weights
      # # Update metrics (includes the metric that tracks the loss)
      # result_metric = 0
      # metric.update_state(y, y_pred)
      # result_metric = metric.result().numpy()
      # # Return a dict mapping metric names to current value
      return gradients, loss.numpy()
    
    def p_train_step(self, x, y, metric, central_server):
      loss_fn = keras.losses.SparseCategoricalCrossentropy()
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
      with tf.GradientTape() as tape:
        y_pred = self.model(x, training=True)  # Forward pass
        y_pred_cet = central_server.model(x, training=True)
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = loss_fn(y, y_pred)
        loss_cet = loss_fn(y, y_pred_cet)
      # Compute gradients
      trainable_vars = self.model.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      # Update weights
      # Update metrics (includes the metric that tracks the loss)
      result_metric = 0
      metric.update_state(y, y_pred_cet)
      result_metric = metric.result().numpy()
      # # Return a dict mapping metric names to current value
      return gradients, loss_cet.numpy(), result_metric
  

def randomize_list(n, p):
  select_list = [0, 1]
  distri = [1-p, p]
  random_list = []
  for i in range(n):
    random_list.append(random.choices(select_list, distri)[0])
  return random_list