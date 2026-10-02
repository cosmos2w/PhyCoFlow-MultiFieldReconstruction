"""Global-distribution components remain nested beneath their family."""

from .cross_copula import CrossJointCopulaCVaR
from .cross_joint import CrossJointTopKSWD
from .mutual_pairwise import MutualPairwiseSWD
from .self_marginal import SelfMarginalW2

__all__ = [
    "CrossJointCopulaCVaR",
    "CrossJointTopKSWD",
    "MutualPairwiseSWD",
    "SelfMarginalW2",
]
