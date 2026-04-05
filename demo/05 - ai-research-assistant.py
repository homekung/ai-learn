"""
Section 2 Project: AI Research Assistant
Complete RAG system with conversation memory
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Data Models
# ============================================================
class ResearchResponse(BaseModel):
    """Structured response from the research assistant."""

    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low based on source quality")
    sources: List[str] = Field(description="List of source documents used")
    key_quotes: List[str] = Field(
        description="Relevant quotes from sources", default=[]
    )
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")


# ============================================================
# Research Assistant Class
# ============================================================


class AIResearchAssistant:
    """AI Research Assistant with document ingestion and retrieval."""

    def __init__(
        self,
        persist_directory: str = "./research_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_directory = persist_directory

        # 1. Embeddings - turns text into vectors
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 2. Splitter - breaks big docs into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # 3. Vector store - stores and searches embeddings
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_docs",
        )

        self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

        print(f"Research Assistant initialized")
        print(f"  Vector store: {persist_directory}")
        print(f"  Documents indexed: {self.vectorstore._collection.count()}")

    def add_documents(
        self,
        documents: List[Document],
        source_name: Optional[str] = None,
    ) -> int:
        """Add documents to the research database."""

        # Tag with source name
        if source_name:
            for doc in documents:
                doc.metadata["source"] = source_name

        # Split into chunks
        chunks = self.splitter.split_documents(documents)

        # Timestamp each chunk
        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()

        # Store in vector DB
        self.vectorstore.add_documents(chunks)

        print(f"Added {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)

    def add_text(self, text: str, source: str, metadata: dict = None) -> int:
        """Add a single text string as a document."""
        doc = Document(
            page_content=text, metadata={"source": source, **(metadata or {})}
        )
        return self.add_documents([doc])
    
    def add_pdf(self, file_path: str) -> int:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        return self.add_documents(documents)

    def _build_retriever(self, use_advanced: bool = False):
        """Build retriever -- basic or advanced"""

        # Base: simple similarity search
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        if not use_advanced:
            return base_retriever

        # Multi-query: LLM generates multiple search queries
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )

        return multi_retriever

    def _format_docs_for_context(self, docs) -> str:
        """Format retrieved documents into a string for the prompt."""
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def get_document_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.vectorstore._collection.count()

    def list_sources(self) -> List[str]:
        """List all unique sources in the database."""
        results = self.vectorstore._collection.get()
        sources = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
        return sorted(list(sources))
    
    def ask_structured(
        self,
        question: str,
        session_id: str = "default",
        use_advanced: bool = True,
    ) -> ResearchResponse:
        """Ask a question and get a structured response."""

        # LLM that returns a Pydantic object instead of a string
        structured_llm = self.llm.with_structured_output(ResearchResponse)

        # Get memory
        history = self._get_session_history(session_id)

        # Retrieve
        retriever = self._build_retriever(use_advanced=use_advanced)
        docs = retriever.invoke(question)
        context = self._format_docs_for_context(docs)
        sources = list(set(d.metadata.get("source", "Unknown") for d in docs))

        # Prompt -- tell the LLM about available sources
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Analyze the provided documents 
    and return a structured response.

    Rules:
    1. ONLY use information from the provided context
    2. If the context doesn't have the answer, say so in the answer field
    3. Set confidence: "high" if directly stated, "medium" if inferred, "low" if partial
    4. Include the source filenames you actually used
    5. Extract key quotes word-for-word from the context
    6. Suggest 2-3 follow-up questions the user might want to ask

    Use conversation history to understand follow-up questions.""",
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context documents:

    {context}

    Available sources: {sources}

    Question: {question}""",
                ),
            ]
        )

        chain = prompt | structured_llm

        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "sources": ", ".join(sources),
                "history": (
                    history.messages[-10:]
                    if hasattr(history, "messages")
                    else history[-10:]
                ),
            }
        )

        # Save to memory (store just the answer text)
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response.answer))

        return response


def print_research_response(question: str, response: ResearchResponse):
    """Pretty print a structured research response."""

    print(f"\nQ: {question}")
    print(f"\n  Answer: {response.answer}")
    print(f"\n  Confidence: {response.confidence}")
    print(f"  Sources: {', '.join(response.sources)}")

    if response.key_quotes:
        print(f"\n  Key Quotes:")
        for q in response.key_quotes:
            print(f'    - "{q}"')

    print(f"\n  Follow-up Questions:")
    for fq in response.follow_up_questions:
        print(f"    - {fq}")


if __name__ == "__main__":
    import shutil

    shutil.rmtree("./research_db", ignore_errors=True)
    assistant = AIResearchAssistant()

    # Add research docs
    assistant.add_text(
        """
        Attention Mechanisms in Neural Networks

        The attention mechanism was introduced in "Attention Is All You Need"
        by Vaswani et al. (2017). It allows models to focus on relevant parts
        of the input when generating output.

        Key concepts:
        - Query, Key, Value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP
        models including BERT, GPT, and T5.
        """,
        source="attention_mechanisms.pdf",
    )

    assistant.add_pdf(
        file_path="./docs/rag_survey.pdf",
    )

    assistant.add_text(
        """
        LangChain and LangGraph Framework Overview

        LangChain is an open-source framework for building LLM applications.
        Key features include modular components, integration with 50+ LLM
        providers, and built-in RAG utilities.

        LangGraph extends LangChain for stateful applications with
        graph-based state management, support for cycles and loops,
        and human-in-the-loop workflows.
        """,
        source="langchain_docs.md",
    )

    session = "structured_demo"

    q1 = "What are the components of RAG?"
    print(f"\nUser: {q1}")
    r1 = assistant.ask_structured(q1, session)
    print_research_response(q1, r1)

    q2 = "How does the second component work?"
    print(f"\n{'- '*30}")
    print(f"\nUser: {q2}")
    r2 = assistant.ask_structured(q2, session)
    print_research_response(q2, r2)

    q3 = "Connect everything we discussed to LangChain."
    print(f"\n{'- '*30}")
    print(f"\nUser: {q3}")
    r3 = assistant.ask_structured(q3, session)
    print_research_response(q3, r3)

    history = assistant._get_session_history(session)
    msg_count = len(history.messages) if hasattr(history, "messages") else len(history)

    print(
        f"""
  ----------------- Final Stat -----------------
  ----------------------------------------------       
  Document ingestion    -> {assistant.get_document_count()} chunks indexed
  Sources tracked       -> {assistant.list_sources()}
  Basic retrieval       -> similarity search
  Advanced retrieval    -> multi-query + compression
  Conversation memory   -> {msg_count} messages in session '{session}'
  Structured output     -> ResearchResponse with {len(ResearchResponse.model_fields)} fields

  From raw text to a production-ready research assistant.
  That's the full RAG pipeline.
    """
    )

    # Cleanup
    shutil.rmtree("./research_db", ignore_errors=True)