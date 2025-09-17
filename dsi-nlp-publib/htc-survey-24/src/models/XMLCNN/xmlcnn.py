from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.torch_train_eval.model import TrainableModel


class XmlCNN(nn.Module, TrainableModel):
    def __init__(self, **config):
        super().__init__()
        vectors = config.get("vectors", None)
        self.output_channel = config["output_channel"]
        target_class = config["num_classes"]
        words_num = config["words_num"]
        words_dim = config["words_dim"]
        self.num_bottleneck_hidden = config["num_bottleneck_hidden"]
        self.dynamic_pool_length = config["dynamic_pool_length"]
        self.ks = 3  # There are three conv nets here
        mode = config.get("mode", "rand")
        dropout = config.get("dropout", 0)
        self._config = config
        del self._config["vectors"]

        self.embed_add = None
        input_channel = 1
        if mode == "rand":
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif mode == "static":
            self.embed = nn.Embedding.from_pretrained(vectors, freeze=True)
        elif mode == "non-static":
            self.embed = nn.Embedding.from_pretrained(vectors, freeze=False)
        elif mode == "multichannel":
            self.embed = nn.Embedding.from_pretrained(vectors, freeze=True)
            self.embed_add = nn.Embedding.from_pretrained(vectors, freeze=False)
            input_channel = 2
        else:
            raise NotImplementedError("Unsupported")

        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (2, words_dim), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (8, words_dim), padding=(7, 0))
        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Adaptive pooling

        self.bottleneck = nn.Linear(self.ks * self.output_channel * self.dynamic_pool_length,
                                    self.num_bottleneck_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, target_class)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, None, None, torch.Tensor]:
        x, y = batch
        x = x.long().to(self.device)
        y = y.long().to(self.device)

        emb_input = self.embed(x)
        if self.embed_add is not None:
            emb_add = self.embed_add(x)
            x = torch.stack([emb_input, emb_add], dim=1)  # (batch, channel_input=2, sent_len, embed_dim)
        else:
            x = emb_input.unsqueeze(1)  # (batch, channel_input=1, sent_len, embed_dim)

        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]

        # (batch, channel_output) * ks
        x = torch.cat(x, 1)  # (batch, channel_output * ks)
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)))
        x = self.dropout(x)
        logits = self.fc1(x)  # (batch, target_size)
        return logits, y, None, None, self.fc1.weight

    def constructor_args(self) -> Dict:
        return self._config

    # def filter_loss_fun_args(self, out: Tuple[torch.Tensor, torch.Tensor]) -> Any:
    #     pred, y = out
    #     if self._multilabel:
    #         # Apparently bce loss wants a float target
    #         y = y.float().to(self.device)
    #     return pred, y
    #
    # def filter_evaluate_fun_args(self, out: Tuple[torch.Tensor, torch.Tensor]) -> Any:
    #     pred, y = out
    #     if self._multilabel:
    #         pred = torch.sigmoid(pred).to(self.device)
    #     else:
    #         pred = F.softmax(pred, dim=1).to(self.device)
    #     return pred, y
