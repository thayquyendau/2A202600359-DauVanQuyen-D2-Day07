from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb
            from chromadb.config import Settings
            
            # Khởi tạo client ChromaDB (mặc định dùng ephemeral store cho nhẹ)
            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=None # We handle embeddings manually
            )
            self._use_chroma = True
        except Exception as e:
            # Fallback về in-memory list nếu không có chromadb hoặc lỗi
            # print(f"ChromaDB not available: {e}. Using in-memory store.")
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Tạo một bản ghi chuẩn hóa để lưu trữ
        # Quan trọng: copy metadata và gán thêm doc_id để sau này dễ xóa (delete)
        metadata = doc.metadata.copy()
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
            
        emb = self._embedding_fn(doc.content)
        
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": emb
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
            
        # 1. Embed query
        query_emb = self._embedding_fn(query)
        
        # 2. Tính similarity với từng record
        results = []
        for rec in records:
            # Dùng dot product (vì _mock_embed trả về normalized vectors hoặc dùng compute_similarity)
            # Ở đây dùng dot product cho nhanh, theo yêu cầu bài lab
            score = _dot(query_emb, rec["embedding"])
            
            res = rec.copy()
            res["score"] = score
            results.append(res)
            
        # 3. Sort theo score giảm dần (cao nhất lên đầu)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.
        """
        for doc in docs:
            record = self._make_record(doc)
            
            if self._use_chroma:
                # Add to ChromaDB collection
                self._collection.add(
                    ids=[f"{doc.id}_{self._next_index}"], # Đảm bảo ID global unique 
                    documents=[record["content"]],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"]]
                )
                self._next_index += 1
            
            # Luôn lưu vào in-memory store để support search linh hoạt
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        # Phân phối sang helper search records
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if not metadata_filter:
            return self.search(query, top_k)
            
        # Lọc (Filter) bằng metadata trước
        filtered_records = []
        for rec in self._store:
            # Kiểm tra xem tất cả keys trong filter có match với metadata của record không
            match = True
            for k, v in metadata_filter.items():
                if rec["metadata"].get(k) != v:
                    match = False
                    break
            if match:
                filtered_records.append(rec)
        
        # Sau đó mới search similarity trên tập đã lọc
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        initial_count = len(self._store)
        
        # Lọc ra những record KHÔNG trùng với doc_id
        self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
        
        # Nếu dùng Chroma, ta cũng cần xóa trong đó (optional trong bài lab này nhưng nên làm)
        if self._use_chroma:
            # ChromaDB support delete by metadata filter
            self._collection.delete(where={"doc_id": doc_id})
            
        return len(self._store) < initial_count
