import numpy as np
import scipy as sp
import scipy.io

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

################################################################################
# Loading the data
################################################################################

import h5py
f = h5py.File('../data/MATS.mat', 'r')

y = np.array(f["y_sim"][:], dtype = np.float64).reshape(-1, 1) * 1e10
y_img = f["y_km"][:].reshape(-1, 1)
m = y.shape[0]

xa = np.array(f["xa"][:], dtype = np.float64).reshape(-1, 1) * 1e5
x_true = f["x_true"][:].reshape(-1, 1) * 1e10
n  = xa.shape[0]

data    = np.array(f["K"]["data"][:], dtype = np.float64)
indices = f["K"]["ir"]
indptr  = f["K"]["jc"]
K = sp.sparse.csc_matrix((data, indices, indptr), shape = (m, n))
K = (K, sp.sparse.csr_matrix(K))

data    = np.array(f["Sainv"]["data"][:] * 1e-20, dtype = np.float64)
indices = f["Sainv"]["ir"]
indptr  = f["Sainv"]["jc"]
SaInv = sp.sparse.csc_matrix((data, indices, indptr), shape = (n, n))

data    = np.array(f["Seinv"]["data"][:] * 1e-20, dtype = np.float64)
indices = f["Seinv"]["ir"]
indptr  = f["Seinv"]["jc"]
SeInv = sp.sparse.csc_matrix((data, indices, indptr), shape = (m, m))

################################################################################
# Running invlib
################################################################################

from invlib.oem           import OEM
from invlib.forward_model import LinearModel
from invlib.vector        import Vector
from invlib.matrix        import Matrix

K     = Matrix(K)
f     = LinearModel(K)
SaInv = Matrix(SaInv)
SeInv = Matrix(SeInv)
xa    = Vector(xa)
y     = Vector(y)

oem = OEM(f, SaInv, SeInv, xa)
oem.optimizer.solver.verbosity = 1
oem.optimizer.solver.tolerance = 1e-4

import time
start_time = time.time()
oem.compute(y)
end_time = time.time()
import os

dt = end_time - start_time
#nt = os.environ["OMP_NUM_THREADS"]
#with open("results_" + str(nt) + ".dat", "w") as f:
#    f.write(str(dt))
