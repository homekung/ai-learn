from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage)
from langchain_core.prompts import (FewShotChatMessagePromptTemplate, MessagesPlaceholder)
import os

load_dotenv()

def create_chat_prompt_template():
    # create chat prompt template
    prompt= ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")

    # format and inspect 
    message = prompt.format_messages(adjective = "funny", topic = "balloon")

    print(message)

def create_multi_message_template():
    # create multi-message template
    prompt = ChatPromptTemplate.from_messages([
                ("system", "your are a helpful assistant that translates {input_language} to {output_language}"),
                ("human", "Tranlate this text: {text}")
            ])
    
    # format and inspect 
    message = prompt.format_messages(input_language = "English", output_language = "French", text = "I love LangChain coding")

    print(message)

def create_message_type():
    messages = [
        SystemMessage(content="You are a robot. Always answer like a pirate"),
        HumanMessage(content="What's the weather today?"),
    ]

    model = init_chat_model("gpt-4o-mini", temperature = 0)
    response = model.invoke(messages)
    print("first answer:")
    print(response.content)

    messages.append(response)
    messages.append(HumanMessage(content="how about tommorow?"))
    response = model.invoke(messages)

    print("second answer:")
    print(response.content)

    print("print message:")
    print(messages)

def create_message_placeholder():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]
    )

    history = [
        HumanMessage(content="My name is Home"),
        AIMessage(content="Nice to meet you, Home!")
    ]

    message = prompt.format_messages(history=history, question="What is my name?")

    model = init_chat_model("gpt-4o-mini", temperature=0)
    response = model.invoke(message)

    print(response.content) 

def create_few_shot_message():
    #prepare example
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"}
    ]

    #prepare example prompt
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])

    #create few shot wrapper.
    fewshot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt = example_prompt
    )

    # create final prompt
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the oposite of each word."),
            fewshot_prompt,
            ("human", "{input}")
        ]
    )

    model = init_chat_model("gpt-4o-mini", temperature=0)
    response = model.invoke(final_prompt.format_messages(input="happy"))
    print(response.content)

def create_reusable_prompt(role: str, question: str):
    system_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}")]
    )

    user_prompt = ChatPromptTemplate.from_messages(
        [("human", "{question}")]
    )

    full_prompt = system_prompt + user_prompt

    model = init_chat_model("gpt-4o-mini", temperature = 0)

    chain = full_prompt | model

    response = chain.invoke({"role": role, "question" : question})
    print(response.content)

if __name__ == "__main__":
    #create_chat_prompt_template()
    #create_multi_message_template()
    #create_message_type()
    create_message_placeholder()
    #create_few_shot_message()
    #create_reusable_prompt("creative", "give me one creative idea.")
    




