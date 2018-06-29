from benchmarks import BenchmarkBase
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

from .cg import fit

import cupy


@sync
@parameterize([('N', [1, 50, 2000]),
               ('max_iter', [5000])])
class CG(BenchmarkBase):
    def setup(self, N, max_iter):
        low, high = -50, 50
        dtype = cupy.float64

        self.A = cupy.random.randint(low, high, size=(N, N))
        self.A = (self.A + self.A.T).astype(dtype)
        x_ans = cupy.random.randint(low, high, size=N).astype(dtype)
        self.b = cupy.dot(self.A, x_ans)
        self.tol = 0.1

    def time_fit(self, N, max_iter):
        fit(self.A, self.b, self.tol, max_iter)
