from typing import List

from langchain.schema import Document


def build_context(docs: List[Document], max_chars: int = 8000):
    # Simple concatenation with source tags
    parts = []
    total = 0
    for i, d in enumerate(docs, 1):
        snippet = d.page_content.strip()
        tag = d.metadata.get("source") or f"doc_{i}"
        block = f"[Source: {tag}]\n{snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)