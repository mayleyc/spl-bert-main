import json
import os
from pathlib import Path
from typing import Tuple, Union, Dict

import torch
import torch.utils.data as td
from tokenizers import Tokenizer
from torchtext.vocab import Vectors


def load_vectors(path: Union[Path, str], oov: bool = True, max_vectors: int = None, write_vocab: bool = True) \
        -> Tuple[Vectors, Dict[str, int]]:
    """
    Load embeddings and create vocabulary file

    :param path: path to embeddings file
    :param oov: whether to add an out-of-vocabulary token and corresponding zero tensor
    :param max_vectors: max num of vectors to load
    :param write_vocab: whether to write a vocabulary file with token to index mapping
    :return: embeddings in torchtext format and the dictionary of tokens to indexes
    """
    name = os.path.basename(path)
    cache = os.path.dirname(path)
    vec = Vectors(name=name, cache=cache, max_vectors=max_vectors)
    if oov:
        # Add element for unknown terms
        vec.vectors = torch.cat([vec.vectors, torch.zeros((1, vec.vectors[0].shape[0]))], dim=0)
        vec.itos.append("<unk>")
        vec.stoi["<unk>"] = len(vec) - 1

    if write_vocab:
        vocab_file = os.path.join(cache, f"{name}.vocab.json")
        with open(vocab_file, "w+") as f:
            json.dump(vec.stoi, f)

    return vec, vec.stoi


class EmbeddingDataset(td.Dataset):
    def __init__(self, x, y, tokenizer: Tokenizer):
        self.tokenizer: Tokenizer = tokenizer
        self.x = tokenizer.encode_batch(x, add_special_tokens=False, is_pretokenized=False)
        self.y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = torch.IntTensor(self.x[index].ids)
        labels = torch.IntTensor(self.y[index])
        return item, labels

    def __len__(self):
        return len(self.x)
