from benchmarks.numpy.common import Benchmark
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

import cupy


black_scholes_kernel = cupy.ElementwiseKernel(
    'T s, T x, T t, T r, T v',  # Inputs
    'T call, T put',  # Outputs
    '''
    const T sqrt_t = sqrt(t);
    const T d1 = (log(s / x) + (r + v * v / 2) * t) / (v * sqrt_t);
    const T d2 = d1 - v * sqrt_t;
    const T cnd_d1 = get_cumulative_normal_distribution(d1);
    const T cnd_d2 = get_cumulative_normal_distribution(d2);
    const T exp_rt = exp(- r * t);
    call = s * cnd_d1 - x * exp_rt * cnd_d2;
    put = x * exp_rt * (1 - cnd_d2) - s * (1 - cnd_d1);
    ''',
    'black_scholes_kernel',
    preamble='''
    __device__
    inline T get_cumulative_normal_distribution(T x) {
        const T A1 = 0.31938153;
        const T A2 = -0.356563782;
        const T A3 = 1.781477937;
        const T A4 = -1.821255978;
        const T A5 = 1.330274429;
        const T RSQRT2PI = 0.39894228040143267793994605993438;
        const T W = 0.2316419;
        const T k = 1 / (1 + W * abs(x));
        T cnd = RSQRT2PI * exp(- x * x / 2) *
            (k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5)))));
        if (x > 0) {
            cnd = 1 - cnd;
        }
        return cnd;
    }
    ''',
)


@sync
@parameterize([('n_options', [1, 10000000])])
class BlackScholes(Benchmark):
    def setup(self, n_options):
        def rand_range(m, M):
            samples = cupy.random.rand(n_options)
            return (m + (M - m) * samples).astype(cupy.float64)
        self.stock_price = rand_range(5, 30)
        self.option_strike = rand_range(1, 100)
        self.option_years = rand_range(0.25, 10)
        self.risk_free = 0.02
        self.volatility = 0.3

    def time_black_scholes(self, n_options):
        s, x, t = self.stock_price, self.option_strike, self.option_years
        r, v = self.risk_free, self.volatility

        sqrt_t = cupy.sqrt(t)
        d1 = (cupy.log(s / x) + (r + v * v / 2) * t) / (v * sqrt_t)
        d2 = d1 - v * sqrt_t

        def get_cumulative_normal_distribution(x):
            A1 = 0.31938153
            A2 = -0.356563782
            A3 = 1.781477937
            A4 = -1.821255978
            A5 = 1.330274429
            RSQRT2PI = 0.39894228040143267793994605993438
            W = 0.2316419

            k = 1 / (1 + W * cupy.abs(x))
            cnd = RSQRT2PI * cupy.exp(-x * x / 2) * (
                k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5)))))
            cnd = cupy.where(x > 0, 1 - cnd, cnd)
            return cnd

        cnd_d1 = get_cumulative_normal_distribution(d1)
        cnd_d2 = get_cumulative_normal_distribution(d2)

        exp_rt = cupy.exp(- r * t)
        call = s * cnd_d1 - x * exp_rt * cnd_d2
        put = x * exp_rt * (1 - cnd_d2) - s * (1 - cnd_d1)
        return call, put

    def time_black_scholes_kernel(self, n_options):
        black_scholes_kernel(
            self.stock_price, self.option_strike, self.option_years,
            self.risk_free, self.volatility)
