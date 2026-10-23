"""Dense retrieval tests — no real Bedrock calls (fake client)."""
from __future__ import annotations

import io
import json

import numpy as np

from rag_eval.bedrock import TitanEmbedder
from rag_eval.chunking import chunk_words
from rag_eval.data import Doc
from rag_eval.retrievers import DenseRetriever, ranked_doc_ids


class FakeBedrock:
    """Minimal stand-in for a bedrock-runtime client.

    Returns a deterministic 3-dim vector derived from the text and counts how
    many times it was actually called, so we can assert the cache works.
    """

    def __init__(self):
        self.calls = 0

    def invoke_model(self, modelId, contentType, accept, body):
        self.calls += 1
        text = json.loads(body)["inputText"]
        # Toy embedding: counts of three marker tokens.
        vec = [
            float(text.lower().count("apple")),
            float(text.lower().count("banana")),
            float(text.lower().count("cluster")),
        ]
        payload = json.dumps({"embedding": vec}).encode()
        return {"body": io.BytesIO(payload)}


def _doc(i, body):
    return Doc(f"d{i}", f"d{i}", body, "src", "2023-01-01", "c", "u")


def test_embedder_caches_and_hits_bedrock_once(tmp_path):
    fake = FakeBedrock()
    emb = TitanEmbedder(client=fake, cache_dir=tmp_path)

    v1 = emb.embed(["apple apple banana"])
    assert fake.calls == 1
    # Same text again -> served from disk cache, no new remote call.
    v2 = emb.embed(["apple apple banana"])
    assert fake.calls == 1
    assert np.array_equal(v1, v2)
    assert v1.tolist() == [[2.0, 1.0, 0.0]]


def test_dense_ranks_semantically_matching_doc_first(tmp_path):
    fake = FakeBedrock()
    emb = TitanEmbedder(client=fake, cache_dir=tmp_path)
    docs = [
        _doc(0, "apple apple apple"),
        _doc(1, "cluster cluster autoscaling cluster"),
        _doc(2, "banana"),
    ]
    r = DenseRetriever(embedder=emb)
    r.index(chunk_words(docs, size=0, prepend_title=False))
    hits = r.retrieve("cluster scaling", 3)
    assert ranked_doc_ids(hits)[0] == "d1"
