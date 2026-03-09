from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

def invoke_chain():
    # Step 1: Define prompt template, model and parser using LCEL
    prompt = ChatPromptTemplate.from_template("Please give me answer in one sentence of this {question}")
    model = ChatOpenAI(model="gpt-4o-mini", 
                        temperature=0.7,
                        max_tokens=1500,
                        timeout=30,
                        max_retries=3)
    parser = StrOutputParser()

    #step 2: Compose with pipe operator (chain)
    chain = prompt | model | parser

    #step 3: execute the chain
    result = chain.invoke({"question": "what is LangChain?"})
    print(f"Response: {result}")

def batch_chain():
    # Step 1: Define prompt template, model and parser using LCEL
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    #step 2: Compose with pipe operator (chain)
    chain = prompt | model | parser

    #step 3: create batch input (dictionary)
    inputs = [
        {"text": "Hellow, how are you?"},
        {"text": "What is your name?"},
        {"text": "where is the nearest restaurant?"}
    ]

    #step 4: execute the batch
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")

def stream_chain():
    # Step 1: Define prompt template, model and parser using LCEL
    prompt = ChatPromptTemplate.from_template("Tell me the story of: {topic}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    #step 2: Compose with pipe operator (chain)
    chain = prompt | model | parser

    #step 3: Print Steam output
    print("Stream Output: ")
    for chunk in chain.stream({"topic": "haiku"}):
        print(chunk, end="", flush=True)

    print()

def basic_invoke_multiple_varaible():
    # Step 1: Define prompt template, model and parser using LCEL
    prompt = ChatPromptTemplate.from_template("give me the marketing tagline of product: {product} and target audience: {audience}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    #step 2: Compose with pipe operator (chain)
    chain = prompt | model | parser

    #step 3: execute
    result = chain.invoke({"product": "AI Course" , "audience": "developers"})

    print(f"Response: {result}")

if __name__ == "__main__":
    invoke_chain()
    batch_chain()
    stream_chain()
    basic_invoke_multiple_varaible()