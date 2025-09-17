import os
import time
from collections import deque
from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
from logzero import logger
# from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.Match.deepxml.evaluation import get_p_1, get_p_3, get_p_5, get_n_3, get_n_5
from src.models.Match.deepxml.optimizers import DenseSparseAdam


# def get_decent_metrics(predictions, targets):
#     return f1_score(targets, predictions, average='macro'), accuracy_score(targets, predictions)


class Model(object):
    def __init__(self, network, model_path, mode, reg=False, hierarchy=None, gradient_clip_value=5.0, device_ids=None,
                 device=None, **kwargs):
        self.device = device
        self.model = nn.DataParallel(network(**kwargs).to(self.device), device_ids=device_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_path, self.state = model_path / "model.pt", {}
        os.makedirs(os.path.split(self.model_path)[0], exist_ok=True)
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.optimizer = None
        self.mlb = kwargs["label_binarizer"]

        self.reg = reg
        if mode == 'train' and reg:
            self.hierarchy = hierarchy
            self.lambda1 = 1e-8
            self.lambda2 = 1e-10

    def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.optimizer.zero_grad()
        self.model.train()
        scores = self.model(train_x)
        loss = self.loss_fn(scores, train_y)

        if self.reg:
            # Output Regularization
            probs = torch.sigmoid(scores)
            regs = torch.zeros(len(probs), len(self.hierarchy)).to(self.device)
            for idx, tup in enumerate(self.hierarchy):
                p = tup[0]
                c = tup[1]
                regs[:, idx] = probs[:, c] - probs[:, p]
            loss += self.lambda1 * torch.sum(nn.functional.relu(regs))  # .item()

            # Parameter Regularization
            # Note: Adding this will make your model training slow
            weights = self.model.module.plaincls.out_mesh_dstrbtn.weight
            regs = torch.zeros(len(weights[0]), len(self.hierarchy)).to(self.device)
            for idx, tup in enumerate(self.hierarchy):
                p = tup[0]
                c = tup[1]
                # DIM: (hidden_size * n_probes, )
                regs[:, idx] = weights[p] - weights[c]
            loss += self.lambda2 * (1 / 2) * torch.norm(regs, p=2) ** 2

        loss.backward()
        self.clip_gradient()
        self.optimizer.step(closure=None)
        return loss.item()

    def predict_step(self, data, k: int):
        self.model.eval()
        with torch.no_grad():
            data_x, labels = data
            infer_time_start: int = time.perf_counter_ns()
            scores = self.model(data_x)
            infer_time_end: int = time.perf_counter_ns()
            infer_time_batch = infer_time_end - infer_time_start
            # scores, labels = torch.topk(self.model(data_x), k)
            return torch.sigmoid(scores).cpu(), labels.cpu(), infer_time_batch

    def get_optimizer(self, **kwargs):
        self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = None,
              nb_epoch=100, step=100, k=5, early=100, verbose=True, swa_warmup=None, **kwargs):
        self.get_optimizer(**({} if opt_params is None else opt_params))
        global_step, best_n5, e = 0, 0.0, 0
        print_loss = 0.0
        for epoch_idx in range(nb_epoch):
            if epoch_idx == swa_warmup:
                self.swa_init()
            for i, (train_x, train_y) in enumerate(train_loader, 1):
                global_step += 1
                loss = self.train_step(train_x, train_y.to(self.device))
                print_loss += loss
                if global_step % step == 0:
                    self.swa_step()
                    self.swap_swa_params()

                    labels = []
                    valid_loss = 0.0
                    self.model.eval()
                    with torch.no_grad():
                        for (valid_x, valid_y) in valid_loader:
                            logits = self.model(valid_x)
                            valid_loss += self.loss_fn(logits, valid_y.to(self.device)).item()
                            scores, tmp = torch.topk(logits, k)
                            labels.append(tmp.cpu())
                    valid_loss /= len(valid_loader)
                    labels = np.concatenate(labels)

                    targets = valid_loader.dataset.data_y
                    # top_1 = self.mlb.transform(self.mlb.classes_[labels[:, :1]])
                    # top_3 = self.mlb.transform(self.mlb.classes_[labels[:, :3]])
                    # top_5 = self.mlb.transform(self.mlb.classes_[labels[:, :5]])
                    # macro_f1, accuracy = get_decent_metrics(labels, targets)
                    p1, p3, p5, n3, n5 = (get_p_1(labels, targets, mlb=None),
                                          get_p_3(labels, targets, mlb=None),
                                          get_p_5(labels, targets, mlb=None),
                                          get_n_3(labels, targets, mlb=None),
                                          get_n_5(labels, targets, mlb=None))
                    if n5 > best_n5:
                        self.save_model(True)
                        best_n5, e = n5, 0
                    else:
                        e += 1
                        if early is not None and e > early:
                            return
                    self.swap_swa_params()
                    if verbose:
                        log_msg = (f'{epoch_idx:d} {i * train_loader.batch_size:d} train loss: '
                                   f'{print_loss / step:.7f} valid loss: {valid_loss:.7f} '
                                   # f'Macro F1: {macro_f1:.5f} Accuracy: {accuracy:.5f}')
                                   f'P@1: {round(p1, 5):.5f} P@3: {round(p3, 5):.5f} P@5: {round(p5, 5):.5f} '
                                   f'N@3: {round(n3, 5):.5f} N@5: {round(n5, 5):.5f} early stop: {e:d}')
                        logger.info(log_msg)
                        print_loss = 0.0

    def predict(self, data_loader: DataLoader, k=100, desc='Predict', **kwargs):
        self.load_model()
        scores_list = list()
        labels_list = list()
        batch_time = .0
        for data_x in tqdm(data_loader, desc=desc, leave=False):
            s, l, t = self.predict_step(data_x, k)
            scores_list.append(s)
            labels_list.append(l)
            batch_time += t
        return np.concatenate(scores_list), np.concatenate(labels_list), \
               batch_time / len(data_loader) / data_loader.batch_size

    def save_model(self, last_epoch):
        if not last_epoch:
            return
        for trial in range(5):
            try:
                torch.save(self.model.module.state_dict(), self.model_path)
                break
            except:
                print('saving failed')

    def load_model(self):
        self.model.module.load_state_dict(torch.load(self.model_path))

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
            max_norm = max_norm.item() if isinstance(max_norm, torch.Tensor) else max_norm
            total_norm = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
            if total_norm > max_norm * self.gradient_clip_value:
                logger.warn(F'Clipping gradients with total norm {round(total_norm, 5)} '
                            F'and max norm {round(max_norm, 5)}')

    def swa_init(self):
        if 'swa' not in self.state:
            logger.info('SWA Initializing')
            swa_state = self.state['swa'] = {'models_num': 1}
            for n, p in self.model.named_parameters():
                swa_state[n] = p.data.cpu().detach()

    def swa_step(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            swa_state['models_num'] += 1
            beta = 1.0 / swa_state['models_num']
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swap_swa_params(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            for n, p in self.model.named_parameters():
                p.data, swa_state[n] = swa_state[n].to(self.device), p.data.cpu()

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']
