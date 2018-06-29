from benchmarks import BenchmarkBase
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

import cupy

from .black_scholes import black_scholes, black_scholes_kernel
from .monte_carlo import compute_option_prices, monte_carlo_kernel


@sync
@parameterize([('n_options', [1, 10000000])])
class BlackScholes(BenchmarkBase):
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
        black_scholes(
            cupy, self.stock_price, self.option_strike,
            self.option_years, self.risk_free, self.volatility)

    def time_black_scholes_kernel(self, n_options):
        black_scholes_kernel(
            self.stock_price, self.option_strike,
            self.option_years, self.risk_free, self.volatility)


@sync
@parameterize([('n_options', [1, 1000]),
               ('n_samples_per_thread', [1, 1000]),
               ('n_threads_per_option', [1, 100000])])
class MonteCarlo(BenchmarkBase):
    def setup(
            self, n_options, n_samples_per_thread, n_threads_per_option):
        def rand_range(m, M):
            samples = cupy.random.rand(n_options)
            return (m + (M - m) * samples).astype(cupy.float64)
        self.stock_price = rand_range(5, 30)
        self.option_strike = rand_range(1, 100)
        self.option_years = rand_range(0.25, 10)
        self.risk_free = 0.02
        self.volatility = 0.3

    def time_monte_carlo(
            self, n_options, n_samples_per_thread, n_threads_per_option):
        compute_option_prices(
            self.stock_price, self.option_strike,
            self.option_years, self.risk_free, self.volatility,
            n_threads_per_option, n_samples_per_thread)


@sync
@parameterize([('gpus', [[0]]),
               ('n_options', [1, 1000]),
               ('n_samples_per_thread', [1, 1000]),
               ('n_threads_per_option', [1, 10000])])
class MonteCarloMultiGPU(BenchmarkBase):
    def setup(
            self, gpus, n_options, n_samples_per_thread, n_threads_per_option):
        def rand_range(m, M):
            samples = cupy.random.rand(n_options)
            return (m + (M - m) * samples).astype(cupy.float64)

        stock_price = rand_range(5, 30)
        option_strike = rand_range(1, 100)
        option_years = rand_range(0.25, 10)
        self.risk_free = 0.02
        self.volatility = 0.3

        self.stock_price_gpus = []
        self.option_strike_gpus = []
        self.option_years_gpus = []
        self.call_prices_gpus = []

        for gpu_id in gpus:
            with cupy.cuda.Device(gpu_id):
                self.stock_price_gpus.append(cupy.array(stock_price))
                self.option_strike_gpus.append(cupy.array(option_strike))
                self.option_years_gpus.append(cupy.array(option_years))
                self.call_prices_gpus.append(cupy.empty(
                    (n_options, n_threads_per_option), dtype=cupy.float64))

    def time_monte_carlo_multigpu(
            self, gpus, n_options, n_samples_per_thread, n_threads_per_option):
        for i, gpu_id in enumerate(gpus):
            with cupy.cuda.Device(gpu_id):
                monte_carlo_kernel(
                    self.stock_price_gpus[i][:, None],
                    self.option_strike_gpus[i][:, None],
                    self.option_years_gpus[i][:, None],
                    self.risk_free, self.volatility, n_samples_per_thread, i,
                    self.call_prices_gpus[i])
