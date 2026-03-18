from typing import Protocol
from jaxtyping import Float, Array

Normalized = Float  # Float in [0, 1]
Standardized = Float  # Float with mean 0 and std 1


class TestFunction(Protocol):
    d: int

    def __call__(self, x: Normalized[Array, "d"]) -> Standardized[Array, ""]: ...
