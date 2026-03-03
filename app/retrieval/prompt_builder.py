"""
prompt_builder.py - Constructs a strict, grounded prompt for the Gemini LLM.
"""

from typing import List, Dict, Any


_SYSTEM_INSTRUCTION = """You are a precise document Q&A assistant for the Swiggy Annual Report.
You must answer questions ONLY using the provided context excerpts from the document.

STRICT RULES:
1. Do NOT infer, guess, or extrapolate beyond the provided context.
2. Do NOT calculate values that are not explicitly written in the context.
3. Do NOT introduce any external knowledge.
4. If the answer is not explicitly present in the context, respond EXACTLY with:
   "The answer is not available in the document."
5. Always cite the page number(s) where you found the information, like: (Page X).
6. Keep your answer factual, concise, and audit-friendly.
"""


def build_prompt(
    query: str,
    context_chunks: List[Dict[str, Any]],
) -> str:
    """Construct the full LLM prompt from retrieved chunks."""
    if not context_chunks:
        context_section = "No relevant context found in the document."
    else:
        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            context_parts.append(
                f"[Excerpt {i} — Page {chunk['page_number']}]\n{chunk['content']}"
            )
        context_section = "\n\n".join(context_parts)

    prompt = f"""{_SYSTEM_INSTRUCTION}

---

CONTEXT FROM SWIGGY ANNUAL REPORT:

{context_section}

---

USER QUESTION:
{query}

ANSWER:"""

    return prompt
