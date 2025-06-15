import sys
from pathlib import Path
import types

import torch

# Stub heavy optional dependencies to avoid import errors
for mod in ["pandas", "matplotlib", "matplotlib.pyplot", "nltk", "nltk.corpus", "nltk.corpus.wordnet", "nltk.corpus.brown"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.GloveEmbedding import glove_embedding


def test_glove_embedding_unknown_word(tmp_path):
    # Create temporary embedding file
    emb_path = tmp_path / "emb.txt"
    with open(emb_path, "w") as f:
        f.write("hello 0.1 0.2 0.3\n")
        f.write("UNK 0.0 0.0 0.0\n")

    vocab_list = ["UNK", "hello"]

    embedding = glove_embedding(path=str(emb_path), embedding_dim=3, vocab_list=vocab_list)

    hello_idx = embedding.word_to_index["hello"]
    unk_idx = embedding.word_to_index["UNK"]

    hello_vec = embedding.embedding_matrix[hello_idx]
    assert torch.allclose(hello_vec, torch.tensor([0.1, 0.2, 0.3]))

    unk_vec = embedding.embedding_matrix[unk_idx]
    assert torch.allclose(unk_vec, torch.tensor([0.0, 0.0, 0.0]))

    unknown_word_idx = embedding.word_to_index.get("world", embedding.word_to_index["UNK"])
    unknown_vec = embedding.embedding_matrix[unknown_word_idx]
    assert torch.allclose(unknown_vec, unk_vec)
