from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os, logging

load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Sample knowledge base
TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

class AdvancedDocumentQA:
    def __init__(self, chroma_dir="./chroma_db"):
        self.chroma_dir = chroma_dir
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)
        self.vector_store = None
        self.retriever = None
        
        # prompt template
        self.prompt = ChatPromptTemplate.from_template(
                """
                Answer the question based on the following context. Be specific and cite which technologies you're referring to.

                Context:
                {context}

                Question: {question}

                Answer:"""
            )

    def init_vector_store(self):
        if os.path.exists(self.chroma_dir):
            self.vector_store = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embedding_model
            )
        else:
            self.vector_store = Chroma.from_documents(
                documents=TECH_DOCS,
                embedding=self.embedding_model,
                persist_directory=self.chroma_dir
            )

    def create_advanced_retriever(self, k=2):
        # Multi-query for better recall
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}), llm=self.llm
        )

        # Compression to focus on relevant info
        compressor = LLMChainExtractor.from_llm(self.llm)

        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=multi_retriever
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
            | StrOutputParser()
        )

        result = self.chain.invoke(question)
        print(f"Question: {question}")
        print(f"A: {result}")



def advanced_rag_workflow():
    documentQa = AdvancedDocumentQA()
    documentQa.init_vector_store()
    documentQa.create_advanced_retriever()
    documentQa.ask("What is the pricing for LangSmith?")

if __name__ == "__main__":
    advanced_rag_workflow()

