import faulthandler
import string
from typing import List, Dict, Tuple

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from tqdm import tqdm

from src.utils.text_utilities.preprocess import text_cleanup, text_cleanup_old

faulthandler.enable()
import os
from pathlib import Path
from logzero import logger

from nltk.data import find
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_dataset(messages: pd.Series, remove_garbage: bool) -> Tuple[List[List[str]], pd.Series]:
    # df:  message, label, sub_labels

    msg = text_cleanup_old(messages, remove_garbage=remove_garbage)
    # 6. Tokenize
    msg_tokens = msg.map(nltk.word_tokenize)
    # 7. Strip punctuation marks
    msg_tokens = msg_tokens.map(lambda tokens: [t.strip(string.punctuation).strip() for t in tokens])

    all_tickets_tokens = msg_tokens.tolist()

    # 8. Join the lists
    tickets_cleaned = pd.Series([" ".join(tokens) for tokens in all_tickets_tokens])
    assert tickets_cleaned.shape[0] == len(all_tickets_tokens), "Noep"

    # assert len(all_tickets_tokens) == len(level_1_labels) == len(level_2_labels) == len(flattened_labels)
    return all_tickets_tokens, tickets_cleaned  # , level_1_labels, level_2_labels, flattened_labels


def process_dataset_new(text: List[str], remove_garbage: bool):
    # df:  message, label, sub_labels

    msg = text_cleanup(text, remove_garbage=remove_garbage)
    # 6. Tokenize
    msg_tokens = map(nltk.word_tokenize, msg)
    # 7. Strip punctuation marks
    msg_tokens = map(lambda tokens: [t.strip(string.punctuation).strip() for t in tokens], msg_tokens)

    return msg_tokens


def create_embedding_model(tickets: List[List[str]], config: Dict) -> Word2Vec:
    logger.info("Training Embeddings...")
    word_2_vec_model = Word2Vec(tickets, min_count=config["min_word_frequency"], vector_size=config["embed_size"],
                                epochs=config["word2vec_epochs"], max_final_vocab=config["max_vocab_size"])
    return word_2_vec_model


def get_data_dump(model_cnf: Dict, df: pd.DataFrame, base_dump: Path) -> Tuple[pd.Series, Word2Vec]:
    dataset_dump, emb_dump = base_dump / "data.jb", base_dump / "vectors.gz"
    if dataset_dump.exists() and emb_dump.exists():
        tickets_tokenized = joblib.load(dataset_dump)
        vectors = Word2Vec.load(str(emb_dump))
    else:
        os.makedirs(base_dump, exist_ok=True)
        tickets = df["message"]
        # PREPROCESS
        tickets_tokenized, tickets = process_dataset(tickets, remove_garbage=model_cnf["remove_garbage"])
        vectors: Word2Vec = create_embedding_model(tickets_tokenized, model_cnf)
        tickets_tokenized = pd.Series(tickets_tokenized)
        joblib.dump(tickets_tokenized, dataset_dump)
        vectors.save(str(emb_dump))
        # np.save(str(emb_dump), emb_init)
    return tickets_tokenized, vectors


def get_data_dump_new(model_cnf: Dict, x: List[str], base_dump: Path, fold: int, _set: str) -> Tuple[
    pd.Series, Word2Vec]:
    dataset_dump, emb_dump = base_dump / f"data_{_set}_fold_{fold}.jb", base_dump / f"vectors_{_set}_fold_{fold}.gz"
    if dataset_dump.exists() and emb_dump.exists():
        text_tokenized = joblib.load(dataset_dump)
        vectors = Word2Vec.load(str(emb_dump))
    else:
        os.makedirs(base_dump, exist_ok=True)
        # PREPROCESS
        text_tokenized = list(process_dataset_new(x, remove_garbage=model_cnf["remove_garbage"]))
        vectors: Word2Vec = create_embedding_model(text_tokenized, model_cnf)
        text_tokenized = pd.Series(text_tokenized)
        joblib.dump(text_tokenized, dataset_dump)
        vectors.save(str(emb_dump))
        # np.save(str(emb_dump), emb_init)
    return text_tokenized, vectors


def _predict(model, loader):
    model.train(False)
    y_pred = list()
    y_true = list()
    with torch.no_grad():
        for i, [x, y] in tqdm(enumerate(loader), total=len(loader)):
            p = model(x)
            pred = p.sigmoid()  # (bs, cat_n)

            # labels = pred.topk(k=2, largest=True, sorted=True, dim=1)[1].detach().cpu().numpy().tolist()
            # labels = ["_".join([str(i) for i in sorted(ls)]) for ls in labels]

            # gt = y.topk(k=2, largest=True, sorted=False, dim=1)[1].detach().cpu().numpy().tolist()
            # gt = ["_".join([str(i) for i in sorted(ls)]) for ls in gt]

            y_pred.append(pred.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

    return np.concatenate(y_pred), np.concatenate(y_true)
