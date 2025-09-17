from typing import Sequence, Optional, List, Dict

import numpy as np
from gensim.models import KeyedVectors
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

TDataX = Sequence[Sequence[str]]
TDataY = Optional[csr_matrix]


def prepare_for_model(texts: List[List[str]], vectors: KeyedVectors, config: Dict):
    max_sentence_len = config["model"]["src_max_len"]

    x_shape = [len(texts), max_sentence_len]

    x = np.zeros(shape=x_shape, dtype="int")
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
    for ticket_index, ticket_text in enumerate(texts):
        word_sequence_index = 0
        for word in ticket_text:
            if word in vectors.key_to_index:
                # index 0 is padding
                x[ticket_index, word_sequence_index] = vectors.key_to_index[word]
                word_sequence_index += 1
                if word_sequence_index == max_sentence_len - 1:
                    break
        # for k in range(word_sequence_index, max_sentence_len):
        #     x[ticket_index, k] = 0
    return x


class MultiLabelDataset(Dataset):
    def __init__(self, data_x: TDataX, data_y: TDataY = None, training=True, **kwargs):
        self.data_y, self.training = data_y, training

        self.data_x = prepare_for_model(data_x, **kwargs)

    def __getitem__(self, item):
        data_x = self.data_x[item]
        if self.training and self.data_y is not None:
            data_y = self.data_y[item].toarray().squeeze(0).astype(
                np.float32)  # .astype(np.float32)  # .toarray().squeeze(0).astype(np.float32)
            return data_x, data_y
        else:
            return data_x

    def __len__(self):
        return len(self.data_x)
