from benchmarks import BenchmarkBase
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

import cupy

from .kmeans import fit


@sync
@parameterize([('n_clusters', [1, 2]),
               ('num', [1, 5000000]),
               ('max_iter', [10]),
               ('use_custom_kernel', [True, False])])
class KMeans(BenchmarkBase):
    def setup(self, n_clusters, num, max_iter, use_custom_kernel):
        samples = cupy.random.randn(num, 2).astype(cupy.float32)
        self.X_train = cupy.r_[samples + 1, samples - 1]

    def time_kmeans(self, n_clusters, num, max_iter, use_custom_kernel):
        fit(self.X_train, n_clusters, max_iter, use_custom_kernel)
