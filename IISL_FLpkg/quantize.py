import random
from numpy.linalg import norm
import tensorflow as tf
import numpy as np
import pdb
import math

def quantize(g, s):
  select_list = [0, 1]
  quan_g = g.copy()
  g_abs = norm(g, 2)
  for i in range(len(g)):
    for l in range(s):
      if(g_abs * (l/s) <= abs(g[i]) and abs(g[i]) < g_abs * (l+1) / s):
          p = (abs(g[i]) / g_abs) * s - l
          distri = [1-p, p]
          l_temp = random.choices(select_list, distri)[0]
          quan_g[i] = (l + l_temp) / s * g_abs
          if(g[i] < 0):
            quan_g[i] = -1 * quan_g[i]
          break
  return quan_g

def quantize_gradient_sum(grd_sum, s, b, p):
  q_grd_sum = [(tf.Variable(grd_sum[i])) for i in range(len(grd_sum))]
  model_params = [None for i in range(len(q_grd_sum))]
  for i in range(len(q_grd_sum)):
    model_params[i] = q_grd_sum[i].numpy().shape
  all_params = []
  for i in range(len(q_grd_sum)):
    all_params = np.append(all_params, q_grd_sum[i] / p) # divided by probability (OFedQIT)
  div_len = math.ceil(len(all_params) / b)  
  for i in range(b):
    temp_params = all_params[i*div_len:(i+1)*div_len]
    temp_params = quantize(temp_params, s)
    all_params[i*div_len:(i+1)*div_len] = temp_params
  q_grd_sum_list = [None for i in range(len(model_params))]
  bound_bef, bound_aft = 0, 0
  for i in range(len(model_params)):
    mulp = 1
    for j in range(len(model_params[i])):
      mulp = mulp * model_params[i][j]
    bound_bef = bound_aft
    bound_aft = bound_aft + mulp
    q_grd_sum_list[i] = all_params[bound_bef:bound_aft].reshape(model_params[i])
  for i in range(len(grd_sum)):
    # Time check
    # work_start = int(time.time() * 1000.0)
    q_grd_sum[i].assign(q_grd_sum_list[i])
  # # Return a dict mapping metric names to current value
  return q_grd_sum