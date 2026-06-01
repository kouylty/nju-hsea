def __getattr__(name):
    if name == "BO":
        from ._bo import BO
        return BO
    elif name == "DropoutAny":
        from ._dropout_any import DropoutAny
        return DropoutAny
    elif name == "RandomGroup":
        from ._random_group import RandomGroup
        return RandomGroup
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

# from ._bo import BO
# from ._dropout_any import DropoutAny
# from ._random_group import RandomGroup
from ._map_elites import MAPElites
from ._ea import EA
from ._bandit_ea import BanditEA
from ._sa import SA
from ._dqn import DQN
from ._random_ps import Random
from ._mcts import MCTS
from ._bops import BOPS
from ._mergebo import MergeBO
from ._utils import from_unit_cube, sobel_sampler, lhs_sampler, permutation_sampler
