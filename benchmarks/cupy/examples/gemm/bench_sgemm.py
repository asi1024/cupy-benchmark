from benchmarks import BenchmarkBase
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

from .utils import benchmark
from .sgemm import sgemm

import cupy


@sync
@parameterize([('m', [1, 1500]),
               ('n', [1, 1500]),
               ('k', [1, 3000])])
class SGEMM(BenchmarkBase):
    def setup(self, m, n, k):
        low, high = -1, 1
        dtype = cupy.float32
        self.A = cupy.random.uniform(low, high, (m, k)).astype(dtype)
        self.B = cupy.random.uniform(low, high, (k, n)).astype(dtype)

    def time_sgemm(self, m, n, k):
        sgemm(self.A, self.B)

    def time_dot(self, m, n, k):
        cupy.dot(self.A, self.B)
