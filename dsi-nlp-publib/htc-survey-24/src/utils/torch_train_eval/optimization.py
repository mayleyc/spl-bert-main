from typing import Sequence

import torch.optim


class MultipleOptimizer:
    """
    Adapter class to support optimization using multiple optimizers.
    Compatible with Trainer for loading and saving states.
    """

    def __init__(self, *op: torch.optim.Optimizer):
        self.optimizers = op

    def zero_grad(self) -> None:
        for op in self.optimizers:
            op.zero_grad()

    def step(self) -> None:
        for op in self.optimizers:
            op.step()

    def __getstate__(self) -> Sequence:
        s = []
        for op in self.optimizers:
            s.append(op.__getstate__())
        return s

    def __setstate__(self, states: Sequence) -> None:
        for op, s in zip(self.optimizers, states):
            op.__setstate__(s)

    def state_dict(self) -> Sequence:
        return self.__getstate__()

    def load_state_dict(self, state: Sequence) -> None:
        self.__setstate__(state)
