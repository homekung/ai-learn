from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
import os

load_dotenv()

def basic_chat_model():

    # Step 1: Define prompt template, model and parser using LCEL
    prompt = ChatPromptTemplate.from_template("Please give me answer in one sentence of this {question}")
    model = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai")
        
    parser = StrOutputParser()

    #step 2: Compose with pipe operator (chain)
    chain = prompt | model | parser

    #step 3: execute the chain
    result = chain.invoke({"question": "what is LangChain?"})
    print(f"Response: {result}")

def basic_switch_model():
    # Step 1: Define prompt template, model and parser using LCEL
    prompt = ChatPromptTemplate.from_template("Please give me answer in one sentence of this {question}")

    if os.getenv("ANTHROPIC_API_KEY"):
        model = init_chat_model(
            model="claude-2",
            model_provider="anthropic")
    else:
        model = init_chat_model(
                model="gpt-4o-mini",
                model_provider="openai")
        
    parser = StrOutputParser()

    #step 2: Compose with pipe operator (chain)
    chain = prompt | model | parser

    #step 3: execute the chain
    result = chain.invoke({"question": "what is LangChain?"})
    print(f"Response: {result}")

def exercise_multi_model():
    
    models = {"gpt-4o-mini": init_chat_model(model="gpt-4o-mini", model_provider="openai"),
              "gpt-4o": init_chat_model(model="gpt-4o", model_provider="openai")}
    
    for key,value in models.items():
        response = value.invoke("What is AI?")
        print(f"Response from {key}: {response.content} ")

def get_response(question: str, model_name: str) -> str:
    model = init_chat_model(model=model_name, temperature= 0.7, streaming=False)
    response = model.invoke(question)
    return response.content

if __name__ == "__main__":
    basic_chat_model()
    basic_switch_model()
    exercise_multi_model()
    print(get_response("What is capital of USA?", "gpt-4o-mini"))