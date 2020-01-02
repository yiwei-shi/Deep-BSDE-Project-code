import numpy as np
import tensorflow.compat.v1 as tf


class Equation(object):
    """Base class for defining BSDE related function."""

    def __init__(self, dim, total_time, num_time_interval, sigma, r, K, x0_range):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample,seed):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the BSDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the BSDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")


class EuropeanCall(Equation):

    def __init__(self, dim, total_time, num_time_interval, sigma, r, K, x0_range):
        super(EuropeanCall, self).__init__(dim, total_time, num_time_interval, sigma, r, K,x0_range)
        self._sigma = sigma
        self._r = r
        self._K = K
        self._x0_range=x0_range
        
    def sample(self, num_sample):
        x_init=self._x0_range[0]+(self._x0_range[1]-self._x0_range[0])*np.random.uniform(0,1,[num_sample,self.dim])
        dw_sample = np.random.randn(num_sample,
                                     self.dim,
                                     self.num_time_interval) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * x_init
        
        factor = np.exp((self._r-(self._sigma**2)/2)*self._delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self._sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample
    
    def f_tf(self, t, x, y, z):
        return -self._r * y

    def g_tf(self, t, x):
        return tf.maximum(x - self._K, 0)