import jax
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import jax.numpy as jnp

from functools import partial
from jax import jit

@jit
def box(x,a,b):
    return jnp.where(x > a, jnp.where(x < b, jnp.ones_like(x), jnp.zeros_like(x)), jnp.zeros_like(x))

@jit
def truncnorm(x, mu, sigma, high, low):
    norm = 2**0.5 / jnp.pi**0.5 / sigma
    norm /= jax.scipy.special.erf((high - mu) / 2**0.5 / sigma) + jax.scipy.special.erf(
        (mu - low) / 2**0.5 / sigma
    )
    prob = jnp.exp(-jnp.power(x - mu, 2) / (2 * sigma**2))
    prob *= norm*box(x, low, high)
    return prob

def compute_bandwidth(X):
    """Uses Scott's rule to compute the bandwidth"""
    d = X.shape[-1]
    N = int(X.size / d)
    bandwidth = np.sqrt(np.diag(np.cov(X.T))) * N**(-1/(d+4))
    return jnp.array(bandwidth)

import numbers

class BoundedKDE:
    def __init__(self, a, b, bandwidth):
        self._a = a
        self._b = b
        self._bandwidth = bandwidth
        if isinstance(bandwidth, numbers.Number):
            self._bandwidth = bandwidth * jnp.ones_like(self._a)
        
    def input_shape(self, evaluation_batch_dimensions=1, data_batch_dimensions=1):
        self.ones_eval = (1,)*evaluation_batch_dimensions
        self.ones_data = (1,)*data_batch_dimensions
        self.d = self._a.size
        self.a = self._a.reshape(self.ones_eval + self.ones_data + (self.d,))
        self.b = self._b.reshape(self.ones_eval + self.ones_data + (self.d,))
        self.bandwidth = self._bandwidth.reshape(self.ones_eval + self.ones_data + (self.d,))
    
    def reshape_tensors(self, X_eval, X_data):
        #print(X_eval.shape, X_data.shape)
        #print(X_eval.shape[:-1] + self.ones_data + (self.d,), self.ones_eval + X_data.shape[:-1] + (self.d,))
        X_eval_reshaped = X_eval.reshape(X_eval.shape[:-1] + self.ones_data    + (self.d,))
        X_data_reshaped = X_data.reshape(self.ones_eval    + X_data.shape[:-1] + (self.d,))
        return X_eval_reshaped, X_data_reshaped
    
    def __call__(self, X_eval, X_data):
        self.input_shape(len(X_eval.shape)-1, len(X_data.shape)-1)
        X_eval_reshaped, X_data_reshaped = self.reshape_tensors(X_eval, X_data)
        #print(X_eval_reshaped.shape, X_data_reshaped.shape)
        #print(self.bandwidth.shape, self.a.shape, self.b.shape)
        result = self.kde(X_eval_reshaped, X_data_reshaped, bandwidth=self.bandwidth, a=self.a, b=self.b)
        result = result.prod(axis=-1).mean(axis=-1)
        return result
    
    def gram_matrix(self, X_data):
        self.input_shape(len(X_data.shape)-1, len(X_data.shape)-1)
        X_eval_reshaped, X_data_reshaped = self.reshape_tensors(X_data, X_data)
        result = self.kde(X_eval_reshaped, X_data_reshaped, bandwidth=self.bandwidth, a=self.a, b=self.b)
        result = result.prod(axis=-1)
        return result
    
    @staticmethod
    @jit
    def kde(X, X_data, bandwidth, a, b):
        return truncnorm(x=X_data, mu=X, sigma=bandwidth, high=b, low=a)