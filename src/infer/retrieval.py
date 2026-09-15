"""Retrieval few-shot cho OCR: nhúng ảnh (CLIP qua transformers), index cosine, cache.

Dùng cho Phase 3: với mỗi ảnh query, chọn K ảnh mẫu gần nhất trong pool train rồi
đưa vào prompt few-shot. KHÔNG cập nhật trọng số model.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

CLIP_NAME = "openai/clip-vit-base-patch32"

try:  # faiss (thư viện ANN nổi tiếng) — tùy chọn; thiếu thì fallback numpy
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - phụ thuộc môi trường
    faiss = None
    _HAS_FAISS = False


def _device():
    """CUDA nếu có, ngược lại CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEmbedder:
    """Nhúng ảnh thành vector bằng CLIP (qua transformers, không cần dep mới)."""

    def __init__(self, model_name=CLIP_NAME, device=None):
        self.model_name = model_name
        self.device = device or _device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed(self, images):
        """``images`` list[PIL] -> ma trận (N, D) đã chuẩn hóa L2."""
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")


class ImageIndex:
    """Embeddings (đã chuẩn hóa L2) + refs (chỉ số dòng dataset gốc).

    Dùng **faiss** (IndexFlatIP, vì vector đã chuẩn hóa L2 nên IP = cosine) nếu
    cài; nếu không thì fallback numpy. Meta (dataset/split/embedder) đi kèm file.
    """

    def __init__(self, embeddings, refs, meta=None, use_faiss=True):
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.refs = np.asarray(refs, dtype="int64")
        self.meta = dict(meta or {})
        self.backend = "numpy"
        self._faiss_index = None
        if use_faiss and _HAS_FAISS and self.embeddings.ndim == 2 and len(self.refs):
            try:
                index = faiss.IndexFlatIP(self.embeddings.shape[1])
                index.add(self.embeddings)
                self._faiss_index = index
                self.backend = "faiss"
            except Exception:  # noqa: BLE001  # pylint: disable=broad-except
                self._faiss_index = None

    def search(self, query_vec, k):
        """Trả về list chỉ số dòng gốc (refs) của ``k`` ảnh gần nhất."""
        if len(self.refs) == 0 or k <= 0:
            return []
        k = min(int(k), len(self.refs))
        query = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        if self._faiss_index is not None:
            _, indices = self._faiss_index.search(query, k)
            return [int(self.refs[i]) for i in indices[0]]
        sims = self.embeddings @ query.reshape(-1)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [int(r) for r in self.refs[top]]

    def save(self, path):
        """Lưu index ra file .npz (kèm meta JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            refs=self.refs,
            meta=np.array(json.dumps(self.meta, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path):
        """Nạp index từ file .npz."""
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        return cls(data["embeddings"], data["refs"], meta)


def build_index(dataset, image_col, embedder, pool=5000, seed=42,
                batch_size=64, progress=True):
    """Chọn ``pool`` ảnh ngẫu nhiên (seeded), nhúng, trả ``ImageIndex``.

    refs giữ chỉ số dòng **gốc** của dataset để eval/lúc chạy lấy lại ảnh + nhãn.
    """
    n = len(dataset)
    pool = n if not pool or pool <= 0 else min(int(pool), n)
    rng = np.random.default_rng(seed)
    refs = rng.permutation(n)[:pool]
    subset = dataset.select(refs.tolist())

    chunks = []
    batch = []
    for i, row in enumerate(subset):
        batch.append(row[image_col])
        if len(batch) >= batch_size:
            chunks.append(embedder.embed(batch))
            batch = []
        if progress and (i + 1) % 500 == 0:
            print(f"  embed {i + 1}/{pool}")
    if batch:
        chunks.append(embedder.embed(batch))

    embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype="float32")
    meta = {"pool": int(pool), "seed": int(seed), "embedder": embedder.model_name}
    return ImageIndex(embeddings, refs, meta)
