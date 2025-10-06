from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from src.utils.torch_train_eval.model import TrainableModel

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torchmetrics as tm

import torch
import torch.nn as nn
import torch.nn.init as init


def mean_emb(emb, mask):
    return (emb * mask).sum(dim=1) / mask.sum(dim=1)  # (bs, dim)


def mean_emb_norm(emb, mask, normalized: bool = False):
    """ Compute the average of embedding, normalizing on the sum of the emb magnitude """
    masked_tensor = emb * mask
    num = masked_tensor.sum(dim=1)
    return num / (num.norm() if normalized else masked_tensor.norm(dim=2).sum())  # (bs, dim)


def min_emb(emb, mask):
    masked_tensor = emb.where(mask > 0, emb.max())
    return masked_tensor.min(dim=1)[0]  # (bs, dim)


def max_emb(emb, mask):
    masked_tensor = emb.where(mask > 0, emb.min())
    return masked_tensor.max(dim=1)[0]  # (bs, dim)


class BERTForClassification(nn.Module, TrainableModel):
    """
    A BERT architecture adapted for sequence classification.
    Averaging strategies inspired from: https://link.springer.com/chapter/10.1007/978-981-15-6168-9_13

    :return Tuple of 4: CLF output, labels, document embeddings w/ dropout, doc embeddings w/o dropout
    """

    def __init__(self, **config):
        super().__init__()
        self.__initial_config = config
        pretrained = config["PRETRAINED_LM"]
        class_n_1 = config["n_class"]
        self.clf_strategy = config["CLF_STRATEGY"]
        self.cfnl: int = config["CLF_STRATEGY_NUM_LAYERS"]
        freeze: bool = config.get("FREEZE_BASE", False)
        add_input_size: int = config.get("additional_input", 0)

        self.device = config.get("device", "cuda:0")

        self.lm = AutoModel.from_pretrained(pretrained, config=config)
        if freeze:
            for param in self.lm.base_model.parameters():
                param.requires_grad = False
        lm_embedding_size = self.lm.config.hidden_size
        if self.clf_strategy in ["concat_cls", "concat_mean_last", "concat_max_last", "max_min_last",
                                 "avg_max_min_last", "max_mean_last", "avg_max_mean_last", "concat_mean_norm",
                                 "concat_mean_normalized", "concat_pool_cls"]:
            lm_embedding_size *= self.cfnl
        # self.pre_classifier = nn.Linear(in_features=lm_embedding_size, out_features=lm_embedding_size)
        self.last_layer_size = lm_embedding_size + add_input_size
        self.clf = nn.Linear(lm_embedding_size + add_input_size, class_n_1)
        conf_bert = self.lm.config
        self.__dropout_p = conf_bert.hidden_dropout_prob if conf_bert.model_type == "bert" else conf_bert.dropout
        self.multilabel = config["multilabel"]

    def forward(self, batch: Tuple[Dict[str, torch.Tensor], torch.LongTensor], *args, **kwargs) -> Any:
        batch, labels = batch
        input_ids = batch["input_ids"].long().to(self.device)
        attention_mask = batch["attention_mask"].float().to(self.device)
        bert_outputs = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        mask = attention_mask.detach().clone().to(self.device)
        # Mask [SEP] token
        mask[torch.arange(0, mask.size(0)).long(), (mask.argmin(dim=1) - 1).clamp(0)] = .0
        # Mask [CLS] token
        mask[:, 0] = .0
        # Adjust dimensions to (bs, seq_len, 1) form multiplication with  (bs, seq_len, dim)
        mask = mask.unsqueeze(-1)

        if self.lm.config.model_type == "bert":
            # (bs, dim), (bs, seq_len, dim), (13, bs, seq_len, dim)
            pooled_output, last_states, hidden_states = bert_outputs.pooler_output, bert_outputs.last_hidden_state, bert_outputs.hidden_states
            if self.clf_strategy == "pool_cls":
                doc_emb = pooled_output  # (bs, dim)
            elif self.clf_strategy == "concat_pool_cls":
                doc_emb = torch.cat([self.lm.pooler(e) for e in hidden_states[-self.cfnl:]], dim=1)
            elif self.clf_strategy == "cls":
                doc_emb = last_states[:, 0]  # (bs, dim)
            elif self.clf_strategy == "concat_cls":
                doc_emb = torch.cat([e[:, 0] for e in hidden_states[-self.cfnl:]], dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_cls":
                doc_emb = torch.stack([e[:, 0] for e in hidden_states[-self.cfnl:]], dim=0).mean(dim=0)  # (bs, dim)
            elif self.clf_strategy == "mean_last":
                # doc_emb = last_states.mean(dim=1)
                doc_emb = mean_emb(last_states, mask)  # (bs, dim)
            elif self.clf_strategy == "concat_mean_last":
                layers_avg = [mean_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_avg, dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_mean_last":
                # layers_avg = [e.mean(dim=1) for e in doc_emb]
                layers_avg = [mean_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, dim)
            elif self.clf_strategy == "max_last":
                doc_emb = max_emb(last_states, mask)  # (bs, dim)
            elif self.clf_strategy == "concat_max_last":
                layers_max = [max_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_max, dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_max_last":
                layers_avg = [max_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, dim)
            # "I'm an artist" tests combining min/max/mean
            elif self.clf_strategy == "max_min_last":
                doc_emb = torch.cat([max_emb(last_states, mask), min_emb(last_states, mask)], dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_max_min_last":
                layers_avg = [torch.cat([max_emb(e, mask), min_emb(e, mask)], dim=1) for e in
                              hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, 2*dim)
            elif self.clf_strategy == "max_mean_last":
                doc_emb = torch.cat([max_emb(last_states, mask), mean_emb(last_states, mask)], dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_max_mean_last":
                layers_avg = [torch.cat([max_emb(e, mask), mean_emb(e, mask)], dim=1) for e in
                              hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, 2*dim)
            elif self.clf_strategy == "mean_norm":
                doc_emb = mean_emb_norm(last_states, mask)  # (bs, dim)
            elif self.clf_strategy == "concat_mean_norm":
                layers_avg = [mean_emb_norm(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_avg, dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "mean_normalized":
                doc_emb = mean_emb_norm(last_states, mask, normalized=True)  # (bs, dim)
            elif self.clf_strategy == "concat_mean_normalized":
                layers_avg = [mean_emb_norm(e, mask, normalized=True) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_avg, dim=1)  # (bs, 2*dim)
            else:
                raise ValueError("Unsupported averaging strategy")
        else:
            raise ValueError("Unsupported LM type")

        add_inputs = kwargs.get("add_clf_inputs", None)
        if add_inputs:
            add_inputs = [a.to(self.device) for a in add_inputs]
            doc_emb = torch.cat([doc_emb, *add_inputs], dim=1).to(self.device)

        drop_doc_emb = F.dropout(doc_emb, p=self.__dropout_p)  # (bs, dim)
        logits = self.clf(drop_doc_emb)
        return logits, labels, drop_doc_emb, doc_emb, self.clf.weight

    def constructor_args(self) -> Dict:
        return self.__initial_config
    
class BERTForClassification_SPL(nn.Module, TrainableModel): 
    """
    A BERT architecture adapted for sequence classification for adding SPL layers.
    Averaging strategies inspired from: https://link.springer.com/chapter/10.1007/978-981-15-6168-9_13

    :return Tuple of 4: CLF output of shape (n, 128), labels, document embeddings w/ dropout, doc embeddings w/o dropout
    """

    def __init__(self, **config):
        super().__init__()
        self.__initial_config = config
        pretrained = config["PRETRAINED_LM"]
        class_n_1 = config["n_class"]
        self.clf_strategy = config["CLF_STRATEGY"]
        self.cfnl: int = config["CLF_STRATEGY_NUM_LAYERS"]
        freeze: bool = config.get("FREEZE_BASE", False)
        add_input_size: int = config.get("additional_input", 0)

        self.device = config.get("device", "cuda:0")

        self.lm = AutoModel.from_pretrained(pretrained, config=config)
        if freeze:
            for param in self.lm.base_model.parameters():
                param.requires_grad = False
        lm_embedding_size = self.lm.config.hidden_size
        if self.clf_strategy in ["concat_cls", "concat_mean_last", "concat_max_last", "max_min_last",
                                 "avg_max_min_last", "max_mean_last", "avg_max_mean_last", "concat_mean_norm",
                                 "concat_mean_normalized", "concat_pool_cls"]:
            lm_embedding_size *= self.cfnl
        # self.pre_classifier = nn.Linear(in_features=lm_embedding_size, out_features=lm_embedding_size)
        self.last_layer_size = lm_embedding_size + add_input_size
        self.clf = nn.Linear(lm_embedding_size + add_input_size, 128) # to append SPL
        conf_bert = self.lm.config
        self.__dropout_p = conf_bert.hidden_dropout_prob if conf_bert.model_type == "bert" else conf_bert.dropout
        self.multilabel = config["multilabel"]

    def forward(self, batch: Tuple[Dict[str, torch.Tensor], torch.LongTensor], *args, **kwargs) -> Any:
        batch, labels = batch
        input_ids = batch["input_ids"].long().to(self.device)
        attention_mask = batch["attention_mask"].float().to(self.device)
        bert_outputs = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        mask = attention_mask.detach().clone().to(self.device)
        # Mask [SEP] token
        mask[torch.arange(0, mask.size(0)).long(), (mask.argmin(dim=1) - 1).clamp(0)] = .0
        # Mask [CLS] token
        mask[:, 0] = .0
        # Adjust dimensions to (bs, seq_len, 1) form multiplication with  (bs, seq_len, dim)
        mask = mask.unsqueeze(-1)

        if self.lm.config.model_type == "bert":
            # (bs, dim), (bs, seq_len, dim), (13, bs, seq_len, dim)
            pooled_output, last_states, hidden_states = bert_outputs.pooler_output, bert_outputs.last_hidden_state, bert_outputs.hidden_states
            if self.clf_strategy == "pool_cls":
                doc_emb = pooled_output  # (bs, dim)
            elif self.clf_strategy == "concat_pool_cls":
                doc_emb = torch.cat([self.lm.pooler(e) for e in hidden_states[-self.cfnl:]], dim=1)
            elif self.clf_strategy == "cls":
                doc_emb = last_states[:, 0]  # (bs, dim)
            elif self.clf_strategy == "concat_cls":
                doc_emb = torch.cat([e[:, 0] for e in hidden_states[-self.cfnl:]], dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_cls":
                doc_emb = torch.stack([e[:, 0] for e in hidden_states[-self.cfnl:]], dim=0).mean(dim=0)  # (bs, dim)
            elif self.clf_strategy == "mean_last":
                # doc_emb = last_states.mean(dim=1)
                doc_emb = mean_emb(last_states, mask)  # (bs, dim)
            elif self.clf_strategy == "concat_mean_last":
                layers_avg = [mean_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_avg, dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_mean_last":
                # layers_avg = [e.mean(dim=1) for e in doc_emb]
                layers_avg = [mean_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, dim)
            elif self.clf_strategy == "max_last":
                doc_emb = max_emb(last_states, mask)  # (bs, dim)
            elif self.clf_strategy == "concat_max_last":
                layers_max = [max_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_max, dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_max_last":
                layers_avg = [max_emb(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, dim)
            # "I'm an artist" tests combining min/max/mean
            elif self.clf_strategy == "max_min_last":
                doc_emb = torch.cat([max_emb(last_states, mask), min_emb(last_states, mask)], dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_max_min_last":
                layers_avg = [torch.cat([max_emb(e, mask), min_emb(e, mask)], dim=1) for e in
                              hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, 2*dim)
            elif self.clf_strategy == "max_mean_last":
                doc_emb = torch.cat([max_emb(last_states, mask), mean_emb(last_states, mask)], dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "avg_max_mean_last":
                layers_avg = [torch.cat([max_emb(e, mask), mean_emb(e, mask)], dim=1) for e in
                              hidden_states[-self.cfnl:]]
                doc_emb = torch.stack(layers_avg, dim=0).mean(dim=0)  # (bs, 2*dim)
            elif self.clf_strategy == "mean_norm":
                doc_emb = mean_emb_norm(last_states, mask)  # (bs, dim)
            elif self.clf_strategy == "concat_mean_norm":
                layers_avg = [mean_emb_norm(e, mask) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_avg, dim=1)  # (bs, 2*dim)
            elif self.clf_strategy == "mean_normalized":
                doc_emb = mean_emb_norm(last_states, mask, normalized=True)  # (bs, dim)
            elif self.clf_strategy == "concat_mean_normalized":
                layers_avg = [mean_emb_norm(e, mask, normalized=True) for e in hidden_states[-self.cfnl:]]
                doc_emb = torch.cat(layers_avg, dim=1)  # (bs, 2*dim)
            else:
                raise ValueError("Unsupported averaging strategy")
        else:
            raise ValueError("Unsupported LM type")

        add_inputs = kwargs.get("add_clf_inputs", None)
        if add_inputs:
            add_inputs = [a.to(self.device) for a in add_inputs]
            doc_emb = torch.cat([doc_emb, *add_inputs], dim=1).to(self.device)

        drop_doc_emb = F.dropout(doc_emb, p=self.__dropout_p)  # (bs, dim)
        logits_for_spl = drop_doc_emb #self.clf(drop_doc_emb)
        return logits_for_spl, labels, drop_doc_emb, doc_emb, self.clf.weight

    def constructor_args(self) -> Dict:
        return self.__initial_config

