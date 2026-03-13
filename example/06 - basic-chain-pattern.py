from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
load_dotenv()

def basic_parallel_chain():
    #create prompt
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarized in one sentence: {text}"
    )

    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the follwing text?: {text}"
    )

    keyword_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords in the following text: {text}\nReturn as a comma-separated list."
    )

    # create model
    model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    # create parser
    parser = StrOutputParser()

    # parallel execution
    analysis_chain = RunnableParallel(
        summary = summarize_prompt | model | parser,
        keywords = keyword_prompt | model | parser,
        sentiment = sentiment_prompt | model | parser,
    )

    
    text = """
    The new AI features are absolutely incredible! Users are loving the
    faster response times and improved accuracy. However, some have noted
    that the pricing could be more competitive. Overall, the product
    launch has been a massive success with record-breaking adoption rates.
    """

    #parallel chain invoke
    results = analysis_chain.invoke({"text": text})
    print("Parallel Analysis Results:")
    print(f"  Summary: {results['summary']}")
    print(f"  Keywords: {results['keywords']}")
    print(f"  Sentiment: {results['sentiment']}")

    
def basic_passthrough_chain():
    #create prompt with context
    prompt = ChatPromptTemplate.from_template(
        "original question: {question}\n"
        "Context: {context} \n\n"
        "Answer the question based on the context"
    )

    # create model
    model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    # create parser
    parser = StrOutputParser()

    # similuatee a retrieve operation
    def fake_retriever(input_dict):
        return " LangChain was created by Harrison Chase in 2022."

    chain = (RunnableParallel(
                    context=RunnableLambda(fake_retriever), question=RunnablePassthrough()
                )
                | RunnableLambda(
                    lambda x: {"context": x["context"], "question" : x["question"]["question"]}
                )
                | prompt
                | model
                | parser
            )
    
    result = chain.invoke({"question": "Who created LangChain?"})
    print(f"Answer: {result}")

def basic_branching_chain():
    # create model
    model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    # Different prompts for different intents
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {input}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    )

    # Classifier
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {input}\nReturn only the classification."
    )

    # Classifier chain
    classifer = classifier_prompt | model | StrOutputParser()

    # Branching chain  based on classification
    def is_code_question(input_dict):
        classification = classifer.invoke(input_dict)
        return "code" in classification.lower()
    
    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser(),  # default branch
    )

    # Test
    questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?",
    ]

    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Q: {q}")
        print(f"A: {result[:100]}...\n")


if __name__ == "__main__":
    basic_parallel_chain()
    basic_passthrough_chain()
    basic_branching_chain()

    




