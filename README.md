# LangChain Learning

This repository contains my experiments and learning notes about **LangChain, LangGraph, and AI agent development**.

The goal of this project is to understand how to build **LLM-powered applications and AI workflows** using modern tools in the LangChain ecosystem.

---

# Topics Covered

This repository includes small examples and experiments related to:

- LangChain Architecture
- Runnable Interface (`invoke`, `batch`, `stream`)
- Chat Models
- Prompt Templates
- Message Types
- Schema and Structured Data
- Chain Pattern and debugging
- RAG - Indexing, Retrieval and Generation
- Indexing - Document Loader, Splitter (Chunk), Embedding and Vector storage
- Vector Store with Chroma, similarity, persist , retriever
- RAG workflow - Multi-Query, Compression, Hybrid, Parent Docs
- Memory - Basic Memory, Multi-user, Trim, Window, Summary and Persistent with SQL
- LangGraph - Basic Node, Edge, State, Condition Edge
- LangGraph - Human in the Loop - Checkpoint, Thread Id

---

# Project Structure
```bash
langchain-learning
│
├─ demo
│ ├─ 01 - movie-review-bot.py
│ ├─ 02 - simple-rag.py
│ ├─ 03 - basic-qa-document.py
│ ├─ 04 - advanced-qa-document.py
│ ├─ 05 - ai-research-assistant.py
│ ├─ 06 - conversation-graph-demo.py
│ └─ 
├─ examples
│ ├─ 01 - basic-runable.py
│ ├─ 02 - basic_chat_model.py
│ ├─ 03 - basic-schema.py
│ ├─ 04 - basic_prompt_message.py
│ ├─ 05 - basic-output-parser.py
│ ├─ 06 - basic-chain-pattern.py
│ ├─ 07 - basic-chain-debug.py
│ ├─ 08 - basic-loader.py
│ ├─ 09 - basic-splitter.py
│ ├─ 10 - basic-embedding-vector.py
│ ├─ 11 - basic-vector-store.py
│ ├─ 12 - basic-rag-workflow.py
│ ├─ 13 - advanced-rag.py
│ ├─ 14 - basic-memory.py
│ ├─ 15 - basic-langgraph.py
│ ├─ 16 - basic-condition-edge.py
│ ├─ 17 - basic-human-interrupt.py
│ └─ 
│
├─ main.py
├─ medium-links.md
├─ README.md
└─ pyproject.toml

```

---

# Setup

Install dependencies using **uv**.

```bash
uv sync
```
or create environment
```bash
uv venv
```

# Environment Variables
Create a .env file:
```bash
OPENAI_API_KEY=your_api_key
```

# Medium Articles
Detailed explanations of these examples are available in my Medium articles.
```bash
medium-links.md
```

# Why This Repository

I am currently learning and exploring how to build:
- LLM applications
- AI agents
- LangChain workflows
- LangGraph pipelines

This repository serves as both:
- a learning log
- a technical reference

# Author

Chanarach Limbanjerdkul
Software Engineer exploring AI Agent Development and LLM Applications.

Medium:
https://medium.com/@chanarachlimbanjerdkul
