import six

from benchmarks.numpy.common import Benchmark
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

import cupy


@sync
@parameterize([('N', [50, 2000]),
               ('max_iter', [50, 5000])])
class CG(Benchmark):
    def setup(self, N, max_iter):
        self.A = cupy.random.randint(-50, 50, size=(N, N))
        self.A = (self.A + self.A.T).astype(cupy.float64)
        x_ans = cupy.random.randint(-50, 50, size=N).astype(cupy.float64)
        self.b = cupy.dot(self.A, x_ans)
        self.tol = 0.1

    def time_fit(self, N, max_iter):
        x = cupy.zeros_like(self.b, dtype=cupy.float64)
        r0 = self.b - cupy.dot(self.A, x)
        p = r0
        for i in six.moves.range(max_iter):
            a = cupy.inner(r0, r0) / cupy.inner(p, cupy.dot(self.A, p))
            x += a * p
            r1 = r0 - a * cupy.dot(self.A, p)
            if cupy.linalg.norm(r1) < self.tol:
                return x
            self.b = cupy.inner(r1, r1) / cupy.inner(r0, r0)
            p = r1 + self.b * p
            r0 = r1
        return x
