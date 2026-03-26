from typing import Protocol
from jaxtyping import Float, Array
from numpy.typing import NDArray as Array


class TestFunction(Protocol):
    def __call__(self, x: Float[Array, "d"]) -> Float[Array, ""]: ...


# 1-dimensional test functions
from .virtual_library import (
    # Many local minima
    GramacyLee,
    # Others
    Forrester,
)

# 2-dimensional test functions
from .virtual_library import (
    # Many local minima
    Bukin6,
    CrossInTray,
    DropWave,
    EggHolder,
    HolderTable,
    Langermann,
    Levy13,
    Schaffer2,
    Schaffer4,
    Shubert,
    # Bowl-shaped
    Bohachevsky1,
    Bohachevsky2,
    Bohachevsky3,
    # Plate-shaped
    Booth,
    Matyas,
    McCormick,
    # Valley-shaped
    Camel3,
    Camel6,
    # Steep Ridges/Drops
    DeJong5,
    Easom,
    # Others
    Beale,
    Branin,
    GoldsteinPrice,
)

# n-dimensional test functions
from .virtual_library import (
    # Many local minima
    Ackley,
    Griewank,
    Levy,
    Rastrigin,
    Schwefel,
    # Bowl-shaped
    Perm0,
    RotatedHyperEllipsoid,
    Sphere,
    SumPowers,
    SumSquares,
    Trid,
    # Plate-shaped
    PowerSum,
    Zakharov,
    # Valley-shaped
    DixonPrice,
    Rosenbrock,
    # Steep Ridges/Drops
    Michalewicz,
    # Others
    Colville,  # only d=4
    Hartmann3,  # only d=3
    Hartmann4,  # only d=4
    Hartmann6,  # only d=6
    Perm,
    Powell,  # only d=4n
    Shekel,  # only d=4
    StyblinskiTang,
)

# game test functions
from .rover import Rover
from .push import Push
from .lunar_lander import LunarLander