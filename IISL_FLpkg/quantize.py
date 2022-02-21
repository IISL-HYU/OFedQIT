import random
from numpy.linalg import norm
import pdb

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