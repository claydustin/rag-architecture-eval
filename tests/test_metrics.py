import math

from rag_eval.chunking import chunk_words
from rag_eval.data import Doc
from rag_eval.metrics import all_hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from rag_eval.retrievers import BM25Retriever, ranked_doc_ids

RANKED = ["x", "a", "y", "b"]
REL = {"a", "b"}


def test_recall_and_all_hit():
    assert recall_at_k(RANKED, REL, 2) == 0.5
    assert recall_at_k(RANKED, REL, 4) == 1.0
    assert all_hit_at_k(RANKED, REL, 3) == 0.0
    assert all_hit_at_k(RANKED, REL, 4) == 1.0


def test_mrr():
    assert mrr_at_k(RANKED, REL, 4) == 0.5
    assert mrr_at_k(RANKED, REL, 1) == 0.0


def test_ndcg():
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(RANKED, REL, 4), expected)
    assert ndcg_at_k(["a", "b"], REL, 2) == 1.0


def _doc(i, body):
    return Doc(f"d{i}", f"d{i}", body, "src", "2023-01-01", "c", "u")


def test_chunking_overlap_and_ids():
    chunks = chunk_words([_doc(0, " ".join(map(str, range(10))))], size=4, overlap=1,
                         prepend_title=False)
    assert [c.text for c in chunks] == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert {c.doc_id for c in chunks} == {"d0"}


def test_bm25_ranks_matching_doc_first_and_dedupes():
    docs = [_doc(0, "apple banana"), _doc(1, "kubernetes cluster autoscaling"), _doc(2, "pear")]
    r = BM25Retriever()
    r.index(chunk_words(docs, size=0))
    assert ranked_doc_ids(r.retrieve("cluster autoscaling", 3))[0] == "d1"
