from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver #langgraph-checkpoint-sqlite
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import tempfile
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def demo_memory_saver():
    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # Keep in process memory
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "inspect-demo"}}

    print("\nState Inspection Demo:\n")

    # Build up some state
    app.invoke({"messages": [HumanMessage(content="Hello!")]}, config)
    app.invoke({"messages": [HumanMessage(content="How are you?")]}, config)

    # Get current state
    state = app.get_state(config)
    print(f"  Message count: {len(state.values['messages'])}")

    # Get state history
    print("\nState history:")
    for i, snapshot in enumerate(app.get_state_history(config)):
        print(f"  Checkpoint {i}: {len(snapshot.values['messages'])} messages")
        if i >= 3:
            print("  ...")
            break

def demo_sqlite_persistence():
    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"\nSQLite Persistence Demo:")
    print(f"Database: {db_path}\n")

    # First session
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content="Remember: The secret code is ALPHA-7")
                ]
            },
            config,
        )
        print(f"Session 1 - Stored secret code")

    # PostgresSaver with a real database!
    # Simulate app restart - new session
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {"messages": [HumanMessage(content="What was the secret code?")]}, config
        )
        print(f"Session 2 - AI: {result['messages'][-1].content}")

def demo_branching_conversations():
    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    # Main conversation
    main_config = {"configurable": {"thread_id": "main"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="What's the weather like in Tokyo on March?")]}, main_config
    )

    print(f"Main: {result['messages'][-1].content[:100]}...")

    # Get checkpoint to branch from
    main_state = app.get_state(main_config)

    # Branch A - Beach vacation
    branch_a_config = {"configurable": {"thread_id": "branch-beach"}}
    # Copy state to new thread
    app.update_state(branch_a_config, main_state.values)

    result_a = app.invoke(
        {"messages": [HumanMessage(content="What about a beach vacation on this weather?")]},
        branch_a_config,
    )
    print(f"Branch A (Beach): {result_a['messages'][-1].content[:100]}...")

    # Branch B - Mountain adventure
    branch_b_config = {"configurable": {"thread_id": "branch-mountain"}}
    app.update_state(branch_b_config, main_state.values)

    result_b = app.invoke(
        {"messages": [HumanMessage(content="What about mountain hiking on this weather?")]},
        branch_b_config,
    )
    print(f"Branch B (Mountain): {result_b['messages'][-1].content[:100]}...")


if __name__ == "__main__":
    demo_memory_saver()
    #demo_sqlite_persistence()
    #demo_branching_conversations()