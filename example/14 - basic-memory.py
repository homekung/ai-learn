from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
import sqlite3
import os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

def basic_memory():
    llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    # Prompt with history placeholder
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # Session storage
    store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Wrap with history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Configuration for this session
    config = {"configurable": {"session_id": "user_123"}}

    # Conversation
    messages = [
        "Hi! My name is Paulo.",
        "I'm learning about LangChain.",
        "What's my name and what am I learning?",
    ]

    print("\nConversation:")
    for msg in messages:
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

    # Show stored history
    print(f"\n--- Stored History ({len(store['user_123'].messages)} messages) ---")
    for msg in store["user_123"].messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:50]}...")

def multi_sessions():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Remember user details."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)
    chain = prompt | llm | StrOutputParser()

    store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    # wrap
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    # Simulate two users
    user_a_config = {"configurable": {"session_id": "user_a"}}
    user_b_config = {"configurable": {"session_id": "user_b"}}

    # User A conversation
    print("\n--- User A ---")
    print("User A: My favorite language is Python")
    resp = chain_with_history.invoke(
        {"input": "My favorite language is Python"}, config=user_a_config
    )
    print(f"AI: {resp}")

    # User B conversation
    print("\n--- User B ---")
    print("User B: I love JavaScript")
    resp = chain_with_history.invoke(
        {"input": "I love JavaScript"}, config=user_b_config
    )
    print(f"AI: {resp}")

    # Ask each user about their preference
    print("\n--- Asking each about their preference ---")

    print("\nUser A: What's my favorite language?")
    resp = chain_with_history.invoke(
        {"input": "What's my favorite language?"}, config=user_a_config
    )
    print(f"AI: {resp}")

    print("\nUser B: What's my favorite language?")
    resp = chain_with_history.invoke(
        {"input": "What's my favorite language?"}, config=user_b_config
    )
    print(f"AI: {resp}")

def basic_message_trimming():

    llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)
    
    # Simulate a long conversation
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(
            content="Python is a high-level programming language known for readability and versatility. It's used in web development, data science, AI, and automation."
        ),
        HumanMessage(content="How do I install it?"),
        AIMessage(
            content="You can install Python from python.org or use package managers like apt, brew, or chocolatey. I recommend Python 3.12+ for new projects."
        ),
        HumanMessage(content="What about pip?"),
        AIMessage(
            content="Pip is Python's package installer. It comes with Python 3.4+. Use 'pip install package_name' to install packages. Consider using virtual environments with venv or uv."
        ),
        HumanMessage(content="Can you summarize everything we discussed?"),
    ]

    print(f"\nOriginal: {len(messages)} messages")

    trimmed = trim_messages(
        messages,
        max_tokens=100,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False
    )

    print(f"After trimming (max 100 tokens): {len(trimmed)} messages")
    print("\nTrimmed messages:")
    for msg in trimmed:
        role = type(msg).__name__.replace("Message", "")
        print(f"  {role}: {msg.content[:60]}...")

def basic_windowed_memory():
    
    # create WindowedChatHistory Class to keep only last k pair
    class WindowedChatHistory(InMemoryChatMessageHistory):
        """Chat history that keeps only last k message pairs."""

        k: int = 2  # Pydantic field - number of exchange pairs to keep

        def add_messages(self, messages):
            super().add_messages(messages)
            # Keep only last k pairs (2k messages: human + ai)
            if len(self.messages) > self.k * 2:
                self.messages = self.messages[-(self.k * 2) :]

    store: Dict[str, WindowedChatHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowedChatHistory(k=2)
        return store[session_id]
    
    llm = init_chat_model(model="gpt-4o-mini",temperature=0.7)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "windowed_test"}}

    # Simulate a conversation with more than 2 pairs
    exchanges = [
        "My name is Paulo",
        "I live in Seattle",
        "I work as an AI engineer",
        "I have 2 cats",
        "What do you remember about me?",
    ]

    print("\nConversation with k=2 window:")
    for i, msg in enumerate(exchanges, 1):
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

        # Show window state after each exchange so students SEE it sliding
        history = store["windowed_test"].messages
        print(f"  [Window: {len(history)} msgs] ", end="")
        facts_in_memory = [
            m.content[:40] for m in history if isinstance(m, HumanMessage)
        ]
        print(f"Remembers: {facts_in_memory}")

def baisc_summary_memory():
    
    # --- Setup Chat ---
    chat_llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    # The conversation prompt: summary of old context + recent messages
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Be concise.\n\n"
                "Summary of earlier conversation:\n{summary}",
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            ("human", "{input}"),
        ]
    )

    chat_chain = chat_prompt | chat_llm | StrOutputParser()

    # --- Setup Summarized ---

    # summary llm no need creativity
    summary_llm = init_chat_model(model="gpt-4o-mini", temperature=0)

    # The summarization prompt: compress messages into a running summary
    summarize_prompt = ChatPromptTemplate.from_template(
        "Condense the current summary and new messages into a single updated summary "
        "(2-3 sentences). Preserve all key facts about the user.\n\n"
        "Current summary:\n{current_summary}\n\n"
        "New messages:\n{new_messages}\n\n"
        "Updated summary:"
    )

    summarize_chain = summarize_prompt | summary_llm | StrOutputParser()

    # --- State ---
    running_summary = ""  # starts empty
    recent_messages = []  # full message objects
    MAX_RECENT = 4  # keep last 4 messages (2 exchanges) before summarizing

    # --- Conversation ---
    exchanges = [
        "My name is Paulo and I'm from Seattle",
        "I work as an AI engineer building RAG systems",
        "I have 2 cats named Luna and Milo",
        "I'm building a LangChain course for Udemy",
        "What do you know about me? List everything.",
    ]

    print(f"\nConfig: keep last {MAX_RECENT} messages, summarize the rest\n")

    for user_input in exchanges:
        print(f"User: {user_input}")

        # 1. Call the LLM with summary + recent messages + new input
        response = chat_chain.invoke(
            {
                "summary": (
                    running_summary if running_summary else "No prior conversation."
                ),
                "recent_messages": recent_messages,
                "input": user_input,
            }
        )
        print(f"AI: {response}")

        # 2. Add this exchange to recent messages
        recent_messages.append(HumanMessage(content=user_input))
        recent_messages.append(AIMessage(content=response))

        # 3. If recent messages exceed limit, summarize the oldest ones
        if len(recent_messages) > MAX_RECENT:
            # Take the oldest messages that will be summarized away
            messages_to_summarize = recent_messages[:-MAX_RECENT]
            formatted = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in messages_to_summarize
            )

            # Update the running summary
            running_summary = summarize_chain.invoke(
                {
                    "current_summary": (
                        running_summary if running_summary else "None yet."
                    ),
                    "new_messages": formatted,
                }
            )

            # Keep only the most recent messages
            recent_messages = recent_messages[-MAX_RECENT:]

            print(
                f"  >>> Summarized! Compressed {len(messages_to_summarize)} old messages"
            )
            print(f"  >>> Summary: {running_summary}")
            print(f"  >>> Recent buffer: {len(recent_messages)} messages")
        print()

    # --- Final state ---
    print("=" * 60)
    print("FINAL MEMORY STATE")
    print("=" * 60)
    print(f"\nRunning summary (compressed old context):\n  {running_summary}")
    print(f"\nRecent messages kept verbatim ({len(recent_messages)}):")
    for msg in recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}")
    print("\nKey insight: ALL facts preserved (name, city, job, cats, course)")
    print("But token cost stays bounded -- old messages are compressed, not deleted!")

def basic_persistent_memory():
    db_path = "./chat_history.db"
    connection_string = f"sqlite:///{db_path}"
    session_id = "persistent_user"

    # Clean slate
    if os.path.exists(db_path):
        os.remove(db_path)

    # helper function
    def build_chain():
        # create llm
        llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

        # create get session from SQLite
        def get_session_hisotry(sid: str) -> BaseChatMessageHistory:
            return SQLChatMessageHistory(
                session_id=sid,
                connection=connection_string
            )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Remember user preferences and facts."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ]
        )

        chain = prompt | llm | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            get_session_hisotry,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    config = {"configurable": {"session_id": session_id}}

    print("\n -------------Run 1 -----------------")

    chain1 = build_chain()

    messages1 = [
        "My name is Paulo. I prefer dark mode themes and Python over JavaScript.",
        "I also like my responses concise -- no fluff.",
    ]
    
    for msg in messages1:
        print(f"User: {msg}")
        response = chain1.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

    # Throw away the chain object entirely -- no in-memory state survives
    del chain1
    
    print("\n -------------Database Checking -----------------")

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT * FROM message_store ORDER BY rowid")
    rows = cursor.fetchall()

    for i, row in enumerate(rows):
        print(f" Row [{i + 1}]: session={row[1]} : {str(row[2])}")

    conn.close()

    print("\n -------------Run 2 -----------------")

    chain2 = build_chain()

    message2 = [
        "What's my name?",
        "What theme do I prefer?",
        "What programming language do I prefer?",
        "How do I like my responses?",
    ]

    for msg in message2:
        print(f"User: {msg}")
        response = chain2.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

    print("\n -------------Database Checking -----------------")

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT * FROM message_store ORDER BY rowid")
    rows = cursor.fetchall()

    for i, row in enumerate(rows):
        print(f" Row [{i + 1}]: session={row[1]} : {str(row[2])}")

    conn.close()


if __name__ == "__main__":
    basic_memory()
    multi_sessions()
    basic_message_trimming()
    basic_windowed_memory()
    baisc_summary_memory()
    basic_persistent_memory()