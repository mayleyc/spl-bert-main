from pathlib import Path

import pandas as pd
import torch
import torch.utils.data as td

from src.utils.text_utilities.preprocess import text_cleanup


class TransformerDatasetFlat(td.Dataset):
    """
    Dataset for Transformers with flattened labels
    """

    def __init__(self, data: pd.Series, labels,
                 remove_garbage: bool = False, multilabel: bool = False, encoder_path: Path = None):
        super().__init__()

        self.multilabel_flag: bool = multilabel
        self.x = data
        self.y = labels
        # self.n_y = len(set([a for b in self.y for a in b])) if self.multilabel_flag else len(set(self.y))
        self.n_y = labels.shape[1]
        self._size = len(self.x)
        self.encoder_path = encoder_path
        self.prepare_dataset(remove_garbage)

    def prepare_dataset(self, remove_garbage: bool = False):
        self.x = list(text_cleanup(self.x, remove_garbage=remove_garbage))
        # self.x = self.x.tolist()

        #  (NOTE): in test, now this has already been done prior.
        # if self.multilabel_flag:
        #     encoder = MultiLabelBinarizer(sparse_output=False)
        #     y = encoder.fit_transform(self.y)
        #     joblib.dump(encoder, self.encoder_path)
        # else:
        #     voc1 = build_vocab_from_iterator([self.y])
        #     y = voc1(self.y)

        self.y = torch.LongTensor(self.y)
        # print(tuple(self.y.size()))

    def __getitem__(self, idx):
        item = self.x[idx]
        label = self.y[idx]
        return item, label

    def __len__(self) -> int:
        return self._size
