from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

# Sample documents
sample_texts = [
        "Python is a versatile programming language used in web development, "
        "data science, machine learning, and automation. It has a simple syntax "
        "that makes it easy to learn and read.",
        "JavaScript is the language of the web. It runs in browsers and on "
        "servers with Node.js. Modern frameworks like React and Vue make "
        "building web applications efficient.",
        "Rust is a systems programming language focused on safety and "
        "performance. It prevents common bugs like null pointer dereferences "
        "and data races at compile time.",
    ]

class SimpleRAG:
    def __init__(self, chroma_dir="./chroma_db"):
        self.chroma_dir = chroma_dir
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self.retriever = None

    def create_chunks(self, texts, chunk_size=500, chunk_overlap=150):
        documents = [
            Document(page_content=t, metadata={"source": i})
            for i, t in enumerate(texts)
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return splitter.split_documents(documents)

    def init_vector_store(self, chunks):
        if os.path.exists(self.chroma_dir):
            self.vector_store = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embedding_model
            )
        else:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.chroma_dir
            )

    def create_retriever(self, k=3):
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            kwargs={"k": k}
        )

    def query(self, question: str):
        docs = self.retriever.invoke(question)

        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.page_content} (source: {doc.metadata['source']})")
            print("-" * 50)


def basic_rag_demo():
    rag = SimpleRAG()
    chunks = rag.create_chunks(sample_texts)
    rag.init_vector_store(chunks)
    rag.create_retriever()
    rag.query("Which programming language is best for frontend web apps?")

if __name__ == "__main__":
    basic_rag_demo()
