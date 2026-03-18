import numpy as np
from .base import TestFunction, Normalized, Standardized, Array


class GoldsteinPrice(TestFunction):
    d = 2

    def __call__(self, x: Normalized[Array, "... d"]) -> Standardized[Array, "..."]:
        # denormalize to [-2, 2]
        x1 = 4 * x[..., 0] - 2
        x2 = 4 * x[..., 1] - 2

        # port of matlab implementation from https://www.sfu.ca/~ssurjano/goldpr.html
        fact1a = (x1 + x2 + 1) ** 2
        fact1b = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        fact1 = 1 + fact1a * fact1b

        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        fact2 = 30 + fact2a * fact2b
        y = fact1 * fact2

        # logarithmic version of the function, standardized
        y = (np.log(y) - 8.6928) / 2.4269
        return y
