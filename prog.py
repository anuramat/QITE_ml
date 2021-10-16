import math
import cmath
import numpy as np

import sys

from testing_routines import *

fl_exact = open(sys.argv[1],"rb")
fl_in = open(sys.argv[2],"rb")

psi_exact = np.load(fl_exact)
psi_in = np.load(fl_in)

Nq = int( math.log2( len(psi_exact) ) )


# ===================================================================
def Rq_vec(tp, theta, q, vec):
  if tp == 2: 
    mask = [i ^ (1 << q) for i in range(len(vec))]
    sgn = np.array([ 1 if (i & (1<<q)) == 0 else 
                    -1 for i in range(len(vec))])

    return math.cos(0.5*theta) * vec \
         + math.sin(0.5*theta) * vec[mask] * sgn[mask]

  if tp == 3:
    z1 = cmath.exp(-1j*0.5*theta)
    z2 = cmath.exp( 1j*0.5*theta)

    return vec * np.array([z1 if (i & (1<<q)) == 0 else 
                           z2 for i in range(len(vec))])
  
  if tp == 1: 
    mask = [i ^ (1 << q) for i in range(len(vec))]

    return math.cos(0.5*theta) * vec \
         - 1j * math.sin(0.5*theta) * vec[mask]

  if tp == 0:
    return vec

  exit("Wrong input for Rq_vec! Abort")
# ===================================================================


# Anzatz - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# --- rot[1](ang_0) --- rot[2](ang_4) ---.-------
#                                        |
# --- rot[1](ang_1) --- rot[2](ang_5) ---x-.-----
#                                          |
# --- rot[1](ang_2) --- rot[2](ang_6) -----x-.---
#                                            |
# --- rot[1](ang_3) --- rot[2](ang_7) -------x---

Nlayers = 2
rot = [2,3]

ang_per_layer = len(rot) * Nq

Nparam = (Nlayers + 1) * ang_per_layer

angs = math.pi*(2*np.random.rand(Nparam)-1)


# construct entangling matrix
cx = np.array([[1,0,0,0],
               [0,0,0,1],
               [0,0,1,0],
               [0,1,0,0]])


Uent = np.identity(2**Nq)
for q in range(Nq-1):
  tmp = np.kron(np.kron( np.identity(2**(Nq-2-q)),
                         cx),
                         np.identity(2**q))

  Uent = tmp.dot(Uent) # The order is important!!!

# convert entangling matrix to mask
ent_mask = Uent.dot( np.array([i for i in range(2**Nq)]) ).astype(int)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Apply anzatz - - - - - - - - - - - - - - - - - - - - - - - - - - -
psi = psi_in.copy()
psi_tst = psi_in.copy()

for ilayer in range(Nlayers):
  for ir,r in enumerate(rot):
    for iq in range(Nq):

      iang = ilayer * ang_per_layer + ir*Nq + iq
      
      psi_tst = Rq_vec_test(r, angs[iang], iq, psi_tst)
      psi = Rq_vec(r, angs[iang], iq, psi)

  if ilayer != Nlayers:
    psi_tst = Uent.dot(psi_tst)
    psi = psi[ent_mask]

print( np.amax(abs(psi_tst - psi)) )

