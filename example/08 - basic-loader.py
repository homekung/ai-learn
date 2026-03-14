from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (TextLoader, WebBaseLoader, DirectoryLoader, PyPDFLoader)
from bs4 import BeautifulSoup
load_dotenv()

def basic_text_loader():
    # create a temporary text file for demo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is the temporary file.\nThis file is used to demo for the Textloader.\n Have the good day"
        )
        temp_file_path = temp_file.name

    try:
        # Load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
             print("Document Content:")
             print(doc.page_content)
             print("Document Metadata:")
             print(doc.metadata)
    finally:
        os.remove(temp_file_path)

def basic_web_loader():
    loader = WebBaseLoader(
        "https://en.wikipedia.org/wiki/Bangkok", bs_kwargs={"parse_only": None}
    )
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from web")
    print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
    print(f"Content length: {len(documents[0].page_content)} characters")
    print(f"Preview: {documents[0].page_content[:200]}...")

def basic_directory_lazy_loader():

    # create temp directory with sample files
    with tempfile.TemporaryDirectory() as tmpdir:
        # create sample file
        for i in range(5):
             path = Path(tmpdir) / f"doc_{i}.txt"
             path.write_text(f"This is docuemnt {i}. It contains sample content.")

        print(tmpdir)
        loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

        for doc in loader.lazy_load():
            print("Document Content Preview:", doc.page_content[:50], "...")
            print("Metadata:", doc.metadata["source"])

def document_structure():
    doc = Document(
        page_content="This is sample content.",
        metadata = {
            "source": "manual.txt",
            "author": "Home",
            "length": 30,
            "tags": ["sample", "test"],
            "created": "2026-03-13"
        }
    )

    print("Document Structure:")
    print(f"  page_content (type): {type(doc.page_content)}")
    print(f"  page_content: {doc.page_content}")
    print(f"  metadata: {doc.metadata}")

def basic_pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    #basic_text_loader()
    #basic_web_loader()
    #basic_directory_lazy_loader()
    #document_structure()
    basic_pdf_loader("./docs/sample.pdf")