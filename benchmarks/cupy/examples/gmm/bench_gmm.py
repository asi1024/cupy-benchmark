from benchmarks import BenchmarkBase
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

import cupy

from .gmm import train_gmm


@sync
@parameterize([('num', [100, 500000]),
               ('dim', [2]),
               ('max_iter', [30])])
class GMM(BenchmarkBase):
    def setup(self, num, dim, max_iter):
        normal = cupy.random.normal
        scale = cupy.ones(dim)
        train1 = normal(1, scale, size=(num, dim)).astype(cupy.float32)
        train2 = normal(-1, scale, size=(num, dim)).astype(cupy.float32)
        mean1 = normal(1, scale, size=dim)
        mean2 = normal(-1, scale, size=dim)
        self.X_train = cupy.r_[train1, train2]
        self.means = cupy.stack([mean1, mean2])
        self.covariances = cupy.random.rand(2, dim)
        self.tol = 0.001

    def time_gmm(self, num, dim, max_iter):
        train_gmm(self.X_train, max_iter, self.tol,
                  self.means, self.covariances)
