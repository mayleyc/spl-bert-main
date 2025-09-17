import logging

import networkx as nx
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn

from src.utils.metrics import label_distance


class BCEChampLoss(nn.Module):
    def __init__(self, g: nx.Graph, enc: MultiLabelBinarizer, beta: float, device: torch.device = None, mode="hard"):
        """
        CHAMP hard loss module, implemented from https://arxiv.org/abs/2206.08653
        """
        super().__init__()
        self.mode = mode

        self.distance_matrix = None
        self.normalised_distance = None
        self.beta = beta
        self._eps = 1.0e-15
        self._n_lab = None
        self.device = device
        self._distance_matrix_(g, enc)
        if mode != "hard":
            logging.warning("Only hard loss is implemented, mode will be set to \"hard\"")

    def _distance_matrix_(self, g, enc: MultiLabelBinarizer) -> None:
        """
        Compute the distance matrix for hard loss

        :param g: label tree structure
        :param enc: binarizer encoder used for labels
        """
        index_to_node = list(g.nodes)
        root_node = None
        if "Root" in index_to_node:
            root_node = "Root"
        elif "root" in index_to_node:
            root_node = "root"
        index_to_node.remove(root_node)
        self._n_lab = len(index_to_node)
        self.distance_matrix = torch.zeros((self._n_lab, self._n_lab), dtype=torch.float).to(self.device)
        for l_i in g.nodes:
            if l_i == root_node:
                continue
            i: int = enc.transform([[l_i]])[0].argmax()
            for l_j in g.nodes:
                if l_j == root_node:
                    continue
                j: int = enc.transform([[l_j]])[0].argmax()
                if j < i:  # Consider distance to be symmetric
                    continue
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = label_distance(g, l_i, l_j)

        max_dist = self.distance_matrix.max()

        assert (self.distance_matrix == self.distance_matrix.transpose(1, 0)).all()

        # Normalization
        self.normalised_distance = (((self.distance_matrix / (max_dist + self._eps)) + 1).pow(2) - 1).to(self.device)

        # np.save("dist.npy", self.distance_matrix.numpy())
        # np.save("norm-dist.npy", self.normalised_distance.numpy())

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # np.save("y-true.npy", target.detach().numpy())
        # np.save("y-pred.npy", prediction.detach().numpy())
        target = target.to(self.device)
        prediction = prediction.to(self.device).sigmoid()
        ones_tensor = torch.ones_like(target, dtype=torch.float).to(self.device)  # (bs, lab)
        distance: torch.Tensor = self.distance_matrix.unsqueeze(0)  # (1, lab, lab)
        max_dist = self.distance_matrix.max()
        zero_f = torch.zeros(1, dtype=torch.float).to(self.device)

        # Add by 1 in order to avoid edge cases of minm distance since distance[i][j] = 0
        distance = torch.where(distance > -1, distance + 1, zero_f)

        # Mask distance matrix by ground truth values
        distance = target.unsqueeze(1) * distance

        # Masked values in above step will be set to 0. In order to compute minm later,
        # we reset those values to a high number greater than max distance
        distance = torch.where(distance < 1., max_dist + 2, distance).float()

        # Setting indices with minm values in a column to 1 and others to 0,
        # such that for row i and column j, if distance[i][j] = 1, then pred label i is mapped to ground truth value j
        distance = torch.where(distance == distance.min(dim=2, keepdim=True)[0], 1, 0)  # (bs, lab, lab)

        # Refill our concerned binarized values (when distance is 1) with their respective normalised distances
        distance = torch.where(distance > 0, self.normalised_distance.unsqueeze(0), zero_f)

        # Modify distance according to how much impact we want from distance penalty in loss calculation
        distance = torch.where(distance != 0., self.beta * distance + 1., zero_f)

        # Computing (1 - p) [part mis-prediction term]
        term1 = (ones_tensor - prediction).unsqueeze(1)
        # Computing log (1-p) [part of mis-prediction term]
        term1 = torch.where(term1 != 0, -torch.log(term1 + self._eps), zero_f)

        # Computing log (p) [part of correct prediction term]
        term2 = torch.where(prediction != 0, -torch.log(prediction + self._eps), zero_f)  # *(alpha)

        # Computing binarized matrix with indices of correct predictions as 1
        correct_ids = target.unsqueeze(1) * torch.eye(self._n_lab).unsqueeze(0).to(self.device)

        # Computing loss
        loss = torch.matmul(term1, distance).squeeze() + torch.matmul(term2.unsqueeze(1), correct_ids).squeeze()

        return loss.sum(1).mean()
