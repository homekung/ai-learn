from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language)
load_dotenv()

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

def basic_recursive_splitter():
    splitter1 = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splitter2 = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    splitter3 = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks1 = splitter1.split_text(SAMPLE_TEXT)
    chunks2 = splitter2.split_text(SAMPLE_TEXT)
    chunks3 = splitter3.split_text(SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"======================Splitter 1 ======================")
    print(f"Number of chunks: {len(chunks1)}")
    print(f"Chunk sizes: {[len(c) for c in chunks1]}")
    print(f"======================Splitter 2 ======================")
    print(f"Number of chunks: {len(chunks2)}")
    print(f"Chunk sizes: {[len(c) for c in chunks2]}")
    print(f"======================Splitter 3 ======================")
    print(f"Number of chunks: {len(chunks3)}")
    print(f"Chunk sizes: {[len(c) for c in chunks3]}")

def basic_overlap():
    text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeated text

    splitter_no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
    )
    splitter_with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )

    chunk_no_overlap = splitter_no_overlap.split_text(text)
    chunk_with_overalp = splitter_with_overlap.split_text(text)

    print("Without overlap:")
    print(f"  Chunk 1 end: ...{chunk_no_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunk_no_overlap[1][:20]}...")

    print("\nWith overlap:")
    print(f"  Chunk 1 end: ...{chunk_with_overalp[0][-20:]}")
    print(f"  Chunk 2 start: {chunk_with_overalp[1][:20]}...")

def basic_markdown_spliter():
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Markdown Splitter produces {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"-------chunk {i}---------")
        print(f" Metadata: {chunk.metadata}")
        print(f" Content: {chunk.page_content[:100]}...")

def basic_code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=500, chunk_overlap=50
    )

    chunks = python_splitter.split_text(SAMPLE_CODE)

    print(f"Code Splitter produce {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} {len(chunk[i])} chars:")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)

def basic_document_splitter():
    loader = PyPDFLoader("./docs/sample.pdf")
    docs = loader.load()

    print(f"Load {len(docs)} documents from PDF")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: {split_docs[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_docs[-1].metadata}")

if __name__ == "__main__":
    basic_recursive_splitter()
    basic_overlap()
    basic_markdown_spliter()
    basic_code_splitter()
    basic_document_splitter()