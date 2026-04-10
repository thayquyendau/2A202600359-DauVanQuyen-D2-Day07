from src.chunking import ChunkingStrategyComparator
from pathlib import Path

text = Path("data/luat_ai_chuong_2.md").read_text(encoding="utf-8")
result = ChunkingStrategyComparator().compare(text, chunk_size=500)

for name, stats in result.items():
    count = stats["count"]
    avg = stats["avg_length"]
    print(f"{name}: count={count}, avg_len={avg:.2f}")
