from typing import Dict, List, Tuple

import numpy as np
from gensim.models import Word2Vec


def train_item2vec_gensim(
    token_sequences: List[List[str]],
    vector_size: int,
    window_size: int,
    negative: int,
    epochs: int,
    min_count: int,
    workers: int,
    seed: int,
    sample: float,
    ns_exponent: float,
    sg: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if not token_sequences:
        raise RuntimeError("Empty corpus for Item2Vec training.")

    model = Word2Vec(
        sentences=token_sequences,
        vector_size=vector_size,
        window=window_size,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        sample=sample,
        ns_exponent=ns_exponent,
        hs=0,
        seed=seed,
        epochs=epochs,
    )

    keys = model.wv.index_to_key
    item_ids = np.array([int(k) for k in keys], dtype=np.int64)
    embeddings = np.stack([model.wv[k] for k in keys], axis=0).astype(np.float32)

    stats = {
        "vocab_size": int(len(keys)),
        "corpus_count": int(model.corpus_count),
        "corpus_total_words": int(model.corpus_total_words),
    }
    return item_ids, embeddings, stats
