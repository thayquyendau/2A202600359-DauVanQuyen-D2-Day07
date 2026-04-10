from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # Lưu lại reference để dùng sau
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieval: Lấy các chunks liên quan nhất từ Vector Store
        results = self.store.search(question, top_k=top_k)
        
        # 2. Augmentation: Gom context lại thành một đoạn văn
        context_parts = []
        for i, res in enumerate(results):
            content = res.get("content", "")
            context_parts.append(f"Chunk {i+1}:\n{content}")
            
        context_str = "\n\n".join(context_parts)
        
        # 3. Generation: Tạo prompt hoàn chỉnh và gọi LLM
        prompt = (
            "Bạn là một trợ lý thông minh. Hãy trả lời câu hỏi dựa TRỰC TIẾP vào context dưới đây.\n"
            "Nếu context không có thông tin, hãy nói 'Tôi không biết'.\n\n"
            f"--- CONTEXT ---\n{context_str}\n\n"
            f"--- QUESTION ---\n{question}\n\n"
            "--- ANSWER ---"
        )
        
        # Gọi "bộ não" (LLM) để lấy câu trả lời cuối cùng
        response = self.llm_fn(prompt)
        return response
