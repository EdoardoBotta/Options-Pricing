import numpy as np
from scipy.special import binom

class BinomialTree:
    def __init__(self, S_0, mu, sigma, r, T, K, steps=5):
        """Initialize Binomial Tree with parameters:
            S_0 : starting stock price
            mu: stock expected return
            sigma: stock volatility
            r: risk-free interest rate
            T: time to expiration
            K: strike price"""

        self.steps = steps
        self.S_0 = S_0
        self.K = K
        self.r = r
        self.T = T
        self.t = T / steps
        
        self.u = np.exp(sigma * np.sqrt(self.t))
        self.d = np.exp(-sigma * np.sqrt(self.t))
        self.p = (np.exp(r * self.t) - self.d) / (self.u - self.d) # risk-neutral probability

    def stock_tree(self):
        n = self.steps + 1
        stock_prices = self.S_0 * np.ones(n)
        u_vec = self.u ** np.arange(n-1, -1, -1)
        d_vec = self.d ** np.arange(0, n, 1)
        final_prices = stock_prices * u_vec * d_vec
        return final_prices
    
    def call_tree(self):
        final_prices = self.stock_tree()
        call_values = np.maximum(np.zeros(self.steps+1), (final_prices - self.K))
        return call_values
    
    def bin_coeff(self):
        p = self.p
        n = self.steps + 1
        coef = np.zeros(n)
        for i in range(n):
            coef[i] = binom(n, i)
        prob_u_vec = p ** np.arange(n-1, -1, -1)
        prob_d_vec = (1-p) ** np.arange(0, n, 1)
        return prob_u_vec * prob_d_vec * coef

    def price_option(self):
        bin_coeff = self.bin_coeff()
        call_values = self.call_tree()
        price = np.exp(-self.r * self.T) * np.sum(call_values * bin_coeff)
        return price
