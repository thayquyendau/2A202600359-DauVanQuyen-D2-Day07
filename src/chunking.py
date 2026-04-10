from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        # Tách câu dựa trên dấu kết thúc câu và khoảng trắng/xuống dòng
        # Regex này chia nhỏ văn bản tại những điểm sau . ! ? nếu có whitespace theo sau
        sentence_pattern = r'(?<=[.!?])\s+'
        raw_sentences = re.split(sentence_pattern, text.strip())
        
        # Làm sạch: loại bỏ khoảng trắng dư thừa ở mỗi câu
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            # Gom đủ "max_sentences_per_chunk" câu thì đóng gói thành 1 chunk
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Nếu vẫn còn sót lại vài câu chưa đủ 1 chunk thì xử lý nốt
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Bắt đầu đệ quy từ list separators mặc định
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: nếu text đã đủ nhỏ rồi thì không cần split thêm nữa
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        # Nếu không còn separator nào để thử (cạn lời) -> đành để nguyên 
        # (hoặc có thể split cưỡng bức bằng fixed size nếu muốn, nhưng ở đây mình stop)
        if not remaining_separators:
            return [current_text]
        
        # Tìm separator hiện tại (cao nhất trong priority list)
        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]
        
        # Split text bằng separator này
        parts = current_text.split(sep)
        
        final_chunks = []
        for part in parts:
            if not part: continue
            
            # Đệ quy cho từng phần nhỏ để đảm bảo mọi phần đều <= chunk_size
            sub_chunks = self._split(part, next_seps)
            final_chunks.extend(sub_chunks)
            
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    # Dot product calculation
    dot_val = _dot(vec_a, vec_b)
    
    # Calculate Magnitudes (L2 Norms)
    mag_a = math.sqrt(sum(x*x for x in vec_a))
    mag_b = math.sqrt(sum(x*x for x in vec_b))
    
    # Check zero-magnitude to avoid ZeroDivisionError (vô nghiệm)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    # Cosine Similarity Formula
    return dot_val / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Khởi tạo các chunkers với cấu hình cơ bản
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        
        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            
            # Tính toán stats cơ bản
            count = len(chunks)
            avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0
            
            comparison[name] = {
                "count": count,
                "avg_length": avg_len,
                "chunks": chunks
            }
            
        return comparison
