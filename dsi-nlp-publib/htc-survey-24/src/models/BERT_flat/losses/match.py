import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Callable


class BCEMatchLoss(nn.Module):
    def __init__(self, hierarchy: List[Tuple[int, int]], base_loss: Callable, device: torch.device = None,
                 out_regularization=True, label_regularization=True):
        """
        MATCH BCE loss module with additional regularization terms
        """
        super().__init__()

        self.device = device
        self.hierarchy = hierarchy
        self.out_reg = out_regularization
        self.label_reg = label_regularization
        self.lambda1 = 1.0e-8
        self.lambda2 = 1.0e-10
        self.base_loss = base_loss

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, _, __, clf_weights: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        loss_clf = self.base_loss(prediction.float().to(self.device), target.float().to(self.device), reduction="mean")
        out_reg = .0
        label_reg = .0

        if self.out_reg:
            # Output Regularization
            probs = prediction.sigmoid().to(self.device)
            regs = torch.zeros(len(probs), len(self.hierarchy)).to(self.device)
            for idx, tup in enumerate(self.hierarchy):
                p = tup[0]
                c = tup[1]
                regs[:, idx] = probs[:, c] - probs[:, p]
            out_reg = self.lambda1 * torch.sum(F.relu(regs))

        if self.label_reg:
            # Parameter Regularization
            regs = torch.zeros(len(clf_weights[0]), len(self.hierarchy)).to(self.device)
            for idx, tup in enumerate(self.hierarchy):
                p = tup[0]
                c = tup[1]
                # DIM: (hidden_size * n_probes, ) = 800
                regs[:, idx] = (clf_weights[p] - clf_weights[c]).to(self.device)
            # Should be self.lambda2 * (1 / 2) * (torch.norm(regs, p=2, dim=0) ** 2).sum() but it is equivalent
            label_reg = self.lambda2 * (1 / 2) * torch.norm(regs, p=2) ** 2

        return loss_clf + out_reg + label_reg
