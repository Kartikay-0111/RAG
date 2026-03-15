"""
query_engine.py — Retrieval query engine using LlamaIndex + Neon pgvector.

Flow:
  1. Ensure global Settings are configured (idempotent via app.llm)
  2. Reconnect to the existing PGVectorStore on Neon (no re-embedding)
  3. Build a VectorStoreIndex from the existing store
  4. Create a QueryEngine with Groq LLM + strict grounded prompt
  5. Call engine.query(question) → cited answer
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.postgres import PGVectorStore

from app.config import (
    NEON_DATABASE_URL,
    EMBEDDING_DIMENSION,
    VECTOR_TABLE_NAME,
    TOP_K_CHUNKS,
    get_async_connection_string,
)
from app.llm import configure_llama_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Strict grounded Q&A prompt ────────────────────────────────────────────────
# Uses LlamaIndex's PromptTemplate so the LLM always stays grounded in
# retrieved chunks.  The prompt enforces citation and refusal behaviour.

_QA_PROMPT_TEMPLATE = PromptTemplate(
    """\
You are a precise document Q&A assistant.
Your task is to answer the user's question using **ONLY** the provided context excerpts.

### DOCUMENT CONTEXT:
---------------------
{context_str}
---------------------

### STRICT RULES:
1.  **Grounding:** Answer **solely** based on the context above. If the context does not contain the specific answer, strictly state: "This information is not available in the provided document." Do not hallucinate or use outside knowledge.
2.  **Accuracy:**
    * **Negative Numbers:** In financial tables, values in brackets `(123.45)` represent negative numbers or losses. Report them clearly as losses or with a negative sign.
    * **Units:** Always preserve the currency and unit (e.g., "INR Millions", "Crores"). Do not convert units unless explicitly asked.
    * **Exactness:** Do not round off numbers. Use the exact precision provided in the text.
3.  **Table Interpretation:**
    * The context may contain tables in Markdown format.
    * Pay close attention to **Column Headers** to ensure you pick data from the correct column/year.
    * Do not confuse "Standalone" figures with "Consolidated" figures if both are present.
4.  **Citation:** End every factual statement with the specific page number from the context in this format: **(Page X)**.
5.  **Document Source:** If the context includes a `document_name` metadata field, mention which document the answer came from when relevant.
6.  **Formatting:** Use bullet points for lists. If comparing two items or periods, present the data in a small Markdown table for clarity.

### USER QUESTION:
{query_str}

### ANSWER:
"""
)


def get_query_engine():
    """
    Build and return a LlamaIndex QueryEngine connected to the existing
    Neon vector store.

    This does NOT re-embed anything.  It loads the VectorStoreIndex from
    the already-populated PGVectorStore and wraps it in a query engine
    with the strict grounded prompt.

    Returns:
        A query engine ready to answer questions.

    Raises:
        RuntimeError: If the vector store connection or index load fails.
    """
    # 1. Ensure global Settings are configured (idempotent)
    configure_llama_settings()

    try:
        # 2. Connect to the existing PGVectorStore on Neon
        vector_store = PGVectorStore.from_params(
            connection_string=NEON_DATABASE_URL,
            async_connection_string=get_async_connection_string(),
            table_name=VECTOR_TABLE_NAME,
            embed_dim=EMBEDDING_DIMENSION,
        )

        # 3. Load the index from the existing store (no re-embedding)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 4. Create the query engine with strict prompt + top-k retrieval
        query_engine = index.as_query_engine(
            similarity_top_k=TOP_K_CHUNKS,
            text_qa_template=_QA_PROMPT_TEMPLATE,
            streaming=False,
        )
    except Exception as exc:
        logger.error(f"Failed to build query engine: {exc}")
        raise RuntimeError(f"Query engine initialisation failed: {exc}") from exc

    logger.info(
        f"QueryEngine ready: top_k={TOP_K_CHUNKS}"
    )
    return query_engine
