"""Amazon Bedrock clients for the dense / agentic phases.

Embeddings go through a thin boto3 wrapper (not langchain-aws) so we own the
on-disk cache and never re-embed unchanged text — a project guardrail. The
cache is keyed by a hash of (model_id, text), so changing the model or the
chunk text invalidates only the affected entries.

Titan Text Embeddings V2 returns a fixed 1024-dim vector per input regardless
of input length.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
# Claude must be invoked via a cross-region inference profile id (the bare
# model id rejects on-demand throughput). Kept here for the agentic phase.
DEFAULT_CHAT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "embed_cache"


def _client(region: str = "us-east-1", profile: str | None = None):
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("bedrock-runtime")


class TitanEmbedder:
    """Cached batch embedder for Titan Text Embeddings V2.

    Each unique (model, text) is embedded once and memoized to disk as a .npy
    vector; subsequent runs (and the dense index build) read from cache.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBED_MODEL,
        region: str = "us-east-1",
        profile: str | None = None,
        cache_dir: Path = CACHE_DIR,
        client=None,
    ):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Lazy client: tests inject a fake; real use builds a boto3 client.
        self._client = client
        self._region = region
        self._profile = profile

    @property
    def client(self):
        if self._client is None:
            self._client = _client(self._region, self._profile)
        return self._client

    def _key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.model_id}\x00{text}".encode()).hexdigest()
        return h

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._key(text)}.npy"

    def _embed_one_remote(self, text: str) -> np.ndarray:
        body = json.dumps({"inputText": text})
        resp = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        payload = json.loads(resp["body"].read())
        return np.asarray(payload["embedding"], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array, using the disk cache where possible.

        Titan's invoke-model embeds one input per call, so "batch" here means we
        iterate but only hit Bedrock for cache misses.
        """
        out: list[np.ndarray] = []
        for t in texts:
            p = self._cache_path(t)
            if p.exists():
                out.append(np.load(p))
                continue
            vec = self._embed_one_remote(t)
            np.save(p, vec)
            out.append(vec)
        return np.vstack(out) if out else np.empty((0, 0), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Single query vector (shape (dim,))."""
        return self.embed([text])[0]
