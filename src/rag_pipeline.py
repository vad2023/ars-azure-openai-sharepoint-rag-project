import logging
import re
from typing import List, Tuple

import tiktoken
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

from .config import Config
from .embedding_index import EmbeddingIndex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def simple_text_from_bytes(filename: str, data: bytes) -> str:
    """Naive text extraction for demo purposes.

    In a real implementation you would plug in proper PDF/DOCX parsers.
    """
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        logger.warning("Failed to decode %s as utf-8", filename)
        return ""


def chunk_text(text: str, max_tokens: int = 512, model_name: str = "gpt-4o") -> List[str]:
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    chunks: List[str] = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks


class AzureOpenAIRag:
    """RAG helper that builds an in-memory index and answers questions."""

    def __init__(self, cfg: Config = Config):
        cfg.validate()
        self.cfg = cfg
        self.client = OpenAIClient(
            endpoint=self.cfg.AOAI_ENDPOINT,
            credential=AzureKeyCredential(self.cfg.AOAI_API_KEY),
        )
        self.index: EmbeddingIndex | None = None
        self._embedding_dim: int | None = None

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        resp = self.client.get_embeddings(
            model=self.cfg.AOAI_EMBED_MODEL,
            input=inputs,
        )
        return [d.embedding for d in resp.data]

    def build_index_for_docs(self, docs: List[Tuple[str, str]]) -> None:
        """Build a vector index for a list of (filename, text) docs."""
        all_chunks: List[str] = []
        meta: List[Tuple[str, str, str]] = []

        for filename, text in docs:
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}::chunk-{idx}"
                all_chunks.append(chunk)
                meta.append((chunk_id, filename, chunk))

        if not all_chunks:
            raise RuntimeError("No content to index")

        embeddings = self._embed(all_chunks)
        dim = len(embeddings[0])
        self._embedding_dim = dim
        self.index = EmbeddingIndex(dim=dim)

        for (chunk_id, filename, chunk), emb in zip(meta, embeddings):
            self.index.add_chunk(chunk_id, filename, chunk, emb)

        logger.info(
            "Indexed %d chunks across %d document(s)",
            len(self.index.chunks),
            len(docs),
        )

    def answer_question(self, question: str, top_k: int = 5) -> str:
        if not self.index:
            raise RuntimeError("Index not built yet")

        q_emb = self._embed([question])[0]
        hits = self.index.search(q_emb, top_k=top_k)

        context_blocks: List[str] = []
        for chunk, score in hits:
            context_blocks.append(
                f"[Score: {score:.3f}, Source: {chunk.source_file}]\n{chunk.text}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are an AI assistant helping USDA ARS Communications and IT teams. "
            "Answer strictly based on the provided context from SharePoint documents. "
            "If something is not covered in the context, say you do not know and "
            "suggest where the answer might live in ARS documentation."
        )

        user_content = f"Question: {question}\n\nContext:\n{context_text}"

        completion = self.client.get_chat_completions(
            model=self.cfg.AOAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )

        return completion.choices[0].message.content
