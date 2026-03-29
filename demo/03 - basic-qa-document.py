#print("welcome to Langchain learning")
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

class RAGResponse(BaseModel):
    """Structured RAG response."""

    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low")
    sources_used: List[str] = Field(description="List of sources referenced")
    follow_up: str = Field(description="Suggested follow-up question")

class DocumentQA:
    def __init__(self, chroma_dir="./chroma_db"):
        self.chroma_dir = chroma_dir
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = init_chat_model(model="gpt-4o-mini", temperature=0.2).with_structured_output(RAGResponse)
        self.vector_store = None
        self.retriever = None
        
        # prompt template
        self.prompt = ChatPromptTemplate.from_template(
        """
            Based on the context below, answer the question.

            Context:
            {context}

            Question: {question}

            Provide a structured response."""
        )

    def init_vector_store(self, texts, chunk_size=500, chunk_overlap=50):
        if os.path.exists(self.chroma_dir):
            self.vector_store = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embedding_model
            )
        else:

            documents = [
                Document(page_content=texts, metadata={"source":"langchain_knowledge_base.md"})
            ]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_documents(documents)

            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.chroma_dir
            )

    def create_retriever(self, k=2):
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            kwargs={"k": k}
        )

    def ask(self, question:str):

        # Format retrieved docs
        def format_docs(docs):
            return "\n\n".join(
                f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
                for doc in docs
            )
        
        # create chain
        self.chain = (
            {
                "context": self.retriever | format_docs, "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )

        result = self.chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Sources: {result.sources_used}")
        print(f"Follow-up: {result.follow_up}")



def basic_rag_workflow():
    documentQa = DocumentQA()
    documentQa.init_vector_store(KNOWLEDGE_BASE)
    documentQa.create_retriever()
    documentQa.ask("What is the pricing for LangSmith?")

if __name__ == "__main__":
    basic_rag_workflow()

