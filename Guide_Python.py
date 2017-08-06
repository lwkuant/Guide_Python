# -*- coding: utf-8 -*-
"""
Guide Python

Record the usages of Python
"""

###
### Use SVD to do dimensionality reduction
###

from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
import numpy as np

r = 100
c = 50
i = 10 # number of dimensionalities to keep

m = np.random.rand(r, c)
m

# Use sklearn to do dimensionality reduction
svd_sklearn = TruncatedSVD(i)
m_sklearn = svd_sklearn.fit_transform(m)
m_sklearn

# Use scipy to do dimensionality reduction
u, s, vt = svd(m)
m_scipy = np.dot(u[:, :i], np.diag(s)[:i, :i])
m_scipy

np.dot(m, vt.T[:, :i])

### Transform new data
m_new = np.random.rand(r - 20, c)
m_new

# Use sklearn
svd_sklearn.transform(m_new)

# Use scipy (Multiply the new data with vt in transposed form reduced to the interested number of dimensionality)
np.dot(m_new, vt.T[:, :i])



