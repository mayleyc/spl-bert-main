from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class TrainableModel(ABC):

    def submodules(self) -> Iterable[Any]:
        """
        Get the list of torch submodules that need to be set the `device` attribute by Trainer.
        The list should not include the model object (self)

        :return: a sequence of references to modules
        """
        return tuple()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def constructor_args(self) -> Dict:
        """
        Get the constructor parameters for this model

        :return: parameters used to instantiate the current class
        """
        pass
