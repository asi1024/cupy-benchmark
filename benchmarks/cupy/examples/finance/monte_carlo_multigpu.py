from benchmarks.numpy.common import Benchmark
from benchmarks.utils import sync
from benchmarks.utils.helper import parameterize

import cupy


monte_carlo_kernel = cupy.ElementwiseKernel(
    'T s, T x, T t, T r, T v, int32 n_samples, int32 seed', 'T call',
    '''
    // We can use special variables i and _ind to get the index of the thread.
    // In this case, we used an index as a seed of random sequence.
    uint64_t rand_state[2];
    init_state(rand_state, i, seed);

    T call_sum = 0;
    const T v_by_sqrt_t = v * sqrt(t);
    const T mu_by_t = (r - v * v / 2) * t;

    // compute the price of the call option with Monte Carlo method
    for (int i = 0; i < n_samples; ++i) {
        const T p = sample_normal(rand_state);
        call_sum += get_call_value(s, x, p, mu_by_t, v_by_sqrt_t);
    }
    // convert the future value of the call option to the present value
    const T discount_factor = exp(- r * t);
    call = discount_factor * call_sum / n_samples;
    ''',
    preamble='''
    typedef unsigned long long uint64_t;

    __device__
    inline T get_call_value(T s, T x, T p, T mu_by_t, T v_by_sqrt_t) {
        const T call_value = s * exp(mu_by_t + v_by_sqrt_t * p) - x;
        return (call_value > 0) ? call_value : 0;
    }

    // Initialize state
    __device__ inline void init_state(uint64_t* a, int i, int seed) {
        a[0] = i + 1;
        a[1] = 0x5c721fd808f616b6 + seed;
    }

    __device__ inline uint64_t xorshift128plus(uint64_t* x) {
        uint64_t s1 = x[0];
        uint64_t s0 = x[1];
        x[0] = s0;
        s1 = s1 ^ (s1 << 23);
        s1 = s1 ^ (s1 >> 17);
        s1 = s1 ^ s0;
        s1 = s1 ^ (s0 >> 26);
        x[1] = s1;
        return s0 + s1;
    }

    // Draw a sample from an uniform distribution in a range of [0, 1]
    __device__ inline T sample_uniform(uint64_t* state) {
        const uint64_t x = xorshift128plus(state);
        // 18446744073709551615 = 2^64 - 1
        return T(x) / T(18446744073709551615);
    }

    // Draw a sample from a normal distribution with N(0, 1)
    __device__ inline T sample_normal(uint64_t* state) {
        T x = sample_uniform(state);
        T s = T(-1.4142135623730950488016887242097);  // = -sqrt(2)
        if (x > 0.5) {
            x = 1 - x;
            s = -s;
        }
        T p = x + T(0.5);
        return s * erfcinv(2 * p);
    }
    ''',
)


@sync
@parameterize([('gpus', [[0]]),
               ('n_options', [1, 1000]),
               ('n_samples_per_thread', [1, 1000]),
               ('n_threads_per_option', [1, 10000])])
class MonteCarloMultiGPU(Benchmark):
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
                self.stock_price_gpus.append(stock_price)
                self.option_strike_gpus.append(option_strike)
                self.option_years_gpus.append(option_years)
                self.call_prices_gpus.append(cupy.empty(
                    (n_options, n_threads_per_option), dtype=cupy.float64))

    def time_compute_option_prices_multigpu(
            self, gpus, n_options, n_samples_per_thread, n_threads_per_option):

        for i, gpu_id in enumerate(gpus):
            with cupy.cuda.Device(gpu_id):
                monte_carlo_kernel(
                    self.stock_price_gpus[i][:, None],
                    self.option_strike_gpus[i][:, None],
                    self.option_years_gpus[i][:, None],
                    self.risk_free, self.volatility,
                    n_samples_per_thread, i, self.call_prices_gpus[i])

        call_mc = cupy.concatenate(self.call_prices_gpus).reshape(
            len(gpus), n_options, n_threads_per_option)
        return call_mc.mean(axis=(0, 2))
