from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# chrom directory
chroma_dir = "./chroma_db"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def create_vector_store():
    # create vector store from documents
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS, 
        embedding=embeddings_model, 
        persist_directory=chroma_dir
    )

    return vectorstore

def basic_chroma(vectorstore):
    #perform similarity search
    query = "What is LangChain?"
    results = vectorstore.similarity_search(query=query, k=2)

    print(f"Top 2 result from query '{query}")
    for i, doc in enumerate(results):
        print(f"Result {i + 1} : {doc.page_content} (Source: {doc.metadata['source']})")


def similarity_search_with_scores(vectorstore):
    #perform similarity search with score
    query = "Explain vector stores."
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f"Top 3 results with scores for query '{query}':")
    for i, (doc, score) in enumerate(results_with_scores):
        final_score = 1 / (1 + score)  # Convert distance to similarity
        print(
            f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
        )


def metadata_filtering(vectorstore):
    query = "What databases are available?"

    # without metadata filtering
    results = vectorstore.similarity_search(query, k=5)
    print(f"Results without metadata filtering for query '{query}':")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    # with metadata filtering
    filter_criteria = {"topic": "database"}
    filtered_results = vectorstore.similarity_search(
        query, k=5, filter=filter_criteria
    )

    print(f"\nResults with metadata filtering for query '{query}':")
    for i, doc in enumerate(filtered_results):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

def persist_chroma(vectorstore):
    original_count = vectorstore._collection.count()
    print(f"Persisted vector store with {original_count} documents.")
    print(f"Vector store persisted at: {chroma_dir}")

    # simulate restart - load from disk
    del vectorstore

    reloaded_vectorstore = Chroma(
        embedding_function=embeddings_model,
        persist_directory=chroma_dir,
    )

    reloaded_count = reloaded_vectorstore._collection.count()
    print(f"Reloaded vector store with {reloaded_count} documents.")

    # verify search still works
    results = reloaded_vectorstore.similarity_search("LangChain", k=2)
    print(f"Search result: {results[0].page_content[:50]}...")

def basic_retriever(vectorstore):

    # basic retriever usage
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    
    docs = retriever.invoke("How do I build AI application?")

    print("Retriever Result")
    for i, doc in enumerate(docs):
        print(f"Result {i+1}: {doc.page_content} (source: {doc.metadata['source']})")

    # MMR retriever
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
    )

    mmr_docs = mmr_retriever.invoke("Vector database and embeddings")

    print("MMR Retriever Result")
    for i, doc in enumerate(mmr_docs):
        print(f"Result {i+1}: {doc.page_content} (source: {doc.metadata['source']})")    


if __name__ == "__main__":
    vectorstore = create_vector_store()
    basic_chroma(vectorstore)
    similarity_search_with_scores(vectorstore)
    metadata_filtering(vectorstore)
    persist_chroma(vectorstore)
    basic_retriever(vectorstore)




