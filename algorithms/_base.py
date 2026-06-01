import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod


class BaseOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def ask(self) -> List[np.ndarray]:
        pass
    
    @abstractmethod
    def tell(self, X: List[np.ndarray], Y: List):
        pass