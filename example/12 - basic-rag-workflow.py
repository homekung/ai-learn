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

chroma_dir = "./chroma_db"
load_dotenv()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)


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


def create_knowledge_base():

    #split the knowledge base to chunks
    splitters = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source":"langchain_knowledge_base.md"})
    chunks = splitters.split_documents([doc])

    # create vector stroe from chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=chroma_dir
    )

    return vector_store

def reload_knowledge_base():
    reloaded_vectorstore = Chroma(
        embedding_function=embeddings_model,
        persist_directory=chroma_dir,
    )

    return reloaded_vectorstore

def basic_simple_rag(question, vector_store):
    
    # create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k":2}
    )
    
    # create prompt template
    prompt = ChatPromptTemplate.from_template(
        """
    Answer the question based only on the following context:

    {context}

    Question: {question}

    Answer (include source for example "(Source: [2] detail.md)"):


    Make sure to answer in a concise manner, 
    and if the answer is not in the context, respond with: "I don't have information about that in my knowledge base."""
    )

    # Format retrieved docs with source
    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    # create chain
    rag_chain =  (
        {"context": retriever | format_docs_with_sources, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")

class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low")
    sources_used: List[str] = Field(description="List of sources referenced")
    follow_up: str = Field(description="Suggested follow-up question")

def basic_structured_rag(question, vector_store):
    
    # create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k":2}
    )
    
    # create prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Based on the context below, answer the question.

        Context:
        {context}

        Question: {question}

        Provide a structured response."""
    )

    # create llm with structure output
    structured_llm = llm.with_structured_output(RAGResponse)

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
            for doc in docs
        )
    
    # create chain
    rag_chain =  (
        {"context": retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    result = rag_chain.invoke(question)
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}")


if __name__ == "__main__":
    vector_store = create_knowledge_base()
    #vector_store = reload_knowledge_base()
    basic_simple_rag("What is LangChain?", vector_store)
    basic_structured_rag("What is the pricing for LangSmith?", vector_store)