# ARS Azure OpenAI + SharePoint RAG Demo

Prototype reference architecture for creating a Retrieval-Augmented Generation (RAG) pattern using:

- SharePoint Online (via Microsoft Graph API)
- Azure OpenAI Service (GPT-4o + Embeddings)
- Python RAG pipeline (chunking, embeddings, vector index, chat completions)

This repo supports:
- Single-site RAG demo
- Multi-site “Copilot-style” agentic routing
- Azure Gov-ready architecture (swap endpoints later)
- Demo CLI for Q&A over SharePoint documents
