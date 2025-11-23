import logging
from dataclasses import dataclass
from typing import Dict, List

from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

from .config import Config
from .embedding_index import EmbeddingIndex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SiteAgent:
    name: str
    description: str
    index: EmbeddingIndex


class CopilotRouter:
    """Router that delegates questions to one or more site-specific RAG indexes."""

    def __init__(self, cfg: Config = Config):
        cfg.validate()
        self.cfg = cfg
        self.client = OpenAIClient(
            endpoint=self.cfg.AOAI_ENDPOINT,
            credential=AzureKeyCredential(self.cfg.AOAI_API_KEY),
        )
        self.agents: Dict[str, SiteAgent] = {}

    def register_agent(self, agent: SiteAgent) -> None:
        self.agents[agent.name] = agent

    def _embed(self, text: str) -> List[float]:
        resp = self.client.get_embeddings(
            model=self.cfg.AOAI_EMBED_MODEL,
            input=[text],
        )
        return resp.data[0].embedding

    def _route(self, question: str) -> List[str]:
        """Use the LLM to decide which knowledge base(s) to use."""
        if not self.agents:
            raise RuntimeError("No agents registered")

        tool_descriptions = "\n".join(
            f"- {name}: {agent.description}"
            for name, agent in self.agents.items()
        )

        system = (
            "You are a router that decides which SharePoint knowledge base(s) are "
            "most relevant for a user question. Return a comma-separated list of "
            "agent names from the provided list. If unsure, choose the single best option."
        )
        user = (
            f"Available knowledge bases:\n{tool_descriptions}\n\n"
            f"User question: {question}\n\n"
            "Which knowledge bases should handle this?"
        )

        resp = self.client.get_chat_completions(
            model=self.cfg.AOAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        logger.info("Router raw output: %s", raw)
        names = [n.strip() for n in raw.split(",") if n.strip() in self.agents]
        if not names:
            # Default to the first registered agent
            names = [next(iter(self.agents.keys()))]
        return names

    def answer(self, question: str, top_k: int = 3) -> str:
        chosen_agents = self._route(question)
        logger.info("Router selected agents: %s", chosen_agents)

        q_emb = self._embed(question)

        context_blocks: List[str] = []
        for name in chosen_agents:
            agent = self.agents[name]
            hits = agent.index.search(q_emb, top_k=top_k)
            for chunk, score in hits:
                context_blocks.append(
                    f"[KB: {name}, Score: {score:.3f}, Source: {chunk.source_file}]\n{chunk.text}"
                )

        context_text = "\n\n---\n\n".join(context_blocks)

        system = (
            "You are an enterprise copilot for USDA ARS. Use the context from multiple "
            "SharePoint knowledge bases to answer the question. If content conflicts, "
            "highlight the discrepancy and prefer more recent or policy-like documents."
        )

        user = f"Question: {question}\n\nContext:\n{context_text}"

        resp = self.client.get_chat_completions(
            model=self.cfg.AOAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
