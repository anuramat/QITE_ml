import math
import cmath
import numpy as np

# ===================================================================
def Rq_vec_test(tp, theta, q, vec):
  Nq = int( math.log2( len(vec) ) )

  if tp == 0:
    return vec

  if tp == 1:
    sqr = np.array([[    math.cos(0.5 * theta),-1j*math.sin(0.5 * theta)],
                    [-1j*math.sin(0.5 * theta),    math.cos(0.5 * theta)]])
    return np.kron(        np.identity(2**(Nq-1-q)), 
                   np.kron(sqr, 
                           np.identity(2**q))).dot(vec)

  if tp == 2:
    sqr = np.array([[math.cos(0.5 * theta),-math.sin(0.5 * theta)],
                    [math.sin(0.5 * theta), math.cos(0.5 * theta)]])
    return np.kron(        np.identity(2**(Nq-1-q)), 
                   np.kron(sqr, 
                           np.identity(2**q))).dot(vec)

  if tp == 3:
    sqr = np.array([[cmath.exp(-1j*0.5*theta),                       0],
                    [                       0, cmath.exp(1j*0.5*theta)]])

    return np.kron(        np.identity(2**(Nq-1-q)), 
                   np.kron(sqr, 
                           np.identity(2**q))).dot(vec)    
  

  exit("Wrong input for Rq_vec_test! Abort")
# ===================================================================
