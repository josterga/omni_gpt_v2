import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

@dataclass
class LazyQueryArtifacts:
    raw_query: str
    _extractor: Optional[Callable[[], List[str]]] = None
    _embedding_builder: Optional[Callable[[], Dict[str, Any]]] = None

    _ngrams: Optional[List[str]] = None
    _chunks: Optional[List[Dict[str, Any]]] = None
    _embedding: Optional[List[float]] = None

    @property
    def ngrams(self) -> List[str]:
        if self._ngrams is None and self._extractor:
            self._ngrams = self._extractor()
        return self._ngrams or []

    @property
    def query_chunks(self) -> List[Dict[str, Any]]:
        if self._chunks is None and self._embedding_builder:
            self._build_embedding()
        return self._chunks or []

    @property
    def query_embedding(self) -> Optional[List[float]]:
        if self._embedding is None and self._embedding_builder:
            self._build_embedding()
        return self._embedding

    def _build_embedding(self):
        if not self._embedding_builder:
            self._chunks, self._embedding = [], None
            return

        # --- SESSION-LEVEL CACHE ---
        if "embedding_cache" not in st.session_state:
            st.session_state["embedding_cache"] = {}

        cache_key = self.raw_query.strip().lower()
        if cache_key in st.session_state["embedding_cache"]:
            cached = st.session_state["embedding_cache"][cache_key]
            self._chunks = cached["chunks"]
            self._embedding = cached["embedding"]
            return

        # Compute and cache
        res = self._embedding_builder()
        self._chunks = res.get("chunks", [])
        self._embedding = res.get("embedding")

        st.session_state["embedding_cache"][cache_key] = {
            "chunks": self._chunks,
            "embedding": self._embedding
        }
