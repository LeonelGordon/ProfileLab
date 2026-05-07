"""RAG persistente con Chroma: sincroniza `data/**/*.md` por hash y recupera por similitud."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from app.rag.loader import chunk_text

COLLECTION_NAME = "profilelab_kb"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PersistentChromaRAG:
    """Índice local en disco; solo re-embeddea archivos nuevos o cuyo contenido cambió."""

    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        root = project_root or _PROJECT_ROOT
        self._data_dir = data_dir or (root / "data")
        self._persist_dir = persist_dir or (root / "vector_store" / "chroma")
        self._model_name = model_name

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = Lock()

    def _get_model(self) -> SentenceTransformer:
        with self._model_lock:
            if self._model is None:
                self._model = SentenceTransformer(self._model_name)
            return self._model

    def _read_markdown_corpus(self) -> Dict[str, Tuple[str, str]]:
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"No existe el directorio de datos: {self._data_dir}")

        files_state: Dict[str, Tuple[str, str]] = {}
        for path in sorted(self._data_dir.rglob("*.md")):
            rel = path.relative_to(self._data_dir).as_posix()
            content = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            files_state[rel] = (digest, content)
        return files_state

    def _stored_hash_by_source(self) -> Dict[str, str]:
        snapshot = self._collection.get(include=["metadatas"])
        out: Dict[str, str] = {}
        for meta in snapshot["metadatas"]:
            if not meta:
                continue
            src = meta["source"]
            if src not in out:
                out[src] = meta["content_hash"]
        return out

    def _delete_source(self, source: str) -> None:
        batch = self._collection.get(where={"source": source}, include=[])
        if batch["ids"]:
            self._collection.delete(ids=batch["ids"])

    def _sync(self) -> None:
        files_state = self._read_markdown_corpus()
        desired_sources = set(files_state.keys())
        stored_hash_by_source = self._stored_hash_by_source()

        for stale in set(stored_hash_by_source.keys()) - desired_sources:
            self._delete_source(stale)

        model = self._get_model()

        for rel, (digest, content) in files_state.items():
            if stored_hash_by_source.get(rel) == digest:
                continue

            self._delete_source(rel)
            chunks = chunk_text(text=content)
            if not chunks:
                continue

            ids = [f"{rel}#{i}" for i in range(len(chunks))]
            embeddings = model.encode(chunks, show_progress_bar=False)
            emb_list = embeddings.tolist()
            metadatas = [
                {"source": rel, "content_hash": digest, "chunk_index": i}
                for i in range(len(chunks))
            ]
            self._collection.add(
                ids=ids,
                embeddings=emb_list,
                documents=chunks,
                metadatas=metadatas,
            )

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not query.strip():
            raise ValueError("La query no puede estar vacía")
        if top_k <= 0:
            raise ValueError("top_k debe ser mayor a 0")

        self._sync()

        if self._collection.count() == 0:
            raise ValueError(
                "No hay chunks indexados. Revisá que exista al menos un .md con texto en data/."
            )

        model = self._get_model()
        vec = model.encode(query, show_progress_bar=False)
        if hasattr(vec, "ndim") and vec.ndim == 1:
            query_embeddings = [vec.tolist()]
        else:
            query_embeddings = vec.tolist()

        res = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents"],
        )
        docs = res["documents"][0] if res.get("documents") else []
        return [d for d in docs if d]


_chroma_singleton: Optional[PersistentChromaRAG] = None
_singleton_lock = Lock()


def get_persistent_chroma_rag() -> PersistentChromaRAG:
    global _chroma_singleton
    with _singleton_lock:
        if _chroma_singleton is None:
            _chroma_singleton = PersistentChromaRAG()
        return _chroma_singleton
