from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import operator

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



class ApprovalState(TypedDict):
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str

def demo_interrupt_for_approval():
    """Interrupt execution for human approval."""

    def create_draft(state: ApprovalState) -> dict:
        response = llm.invoke(f"Create a professional response for: {state['request']}")
        return {"draft": response.content}

    def wait_for_approval(state: ApprovalState) -> dict:
        # This node is where we'll interrupt
        return state

    def finalize(state: ApprovalState) -> dict:
        if state["approved"]:
            return {"final": state["draft"]}
        else:
            response = llm.invoke(
                f"Revise this draft based on feedback:\n\n"
                f"Draft: {state['draft']}\n\n"
                f"Feedback: {state['feedback']}"
            )
            return {"final": response.content}

    graph = StateGraph(ApprovalState)

    graph.add_node("draft", create_draft)
    graph.add_node("approval", wait_for_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)

    # Compile with checkpointer and interrupt
    memory = MemorySaver()
    app = graph.compile(
        checkpointer=memory, interrupt_before=["approval"]  # Pause before this node
    )

    # Configuration for this thread
    config = {"configurable": {"thread_id": "demo-1"}}

    # ─── PHASE 1: Run until interrupt ───
    result = app.invoke(
        {
            "request": "Write a thank-you email for a job interview",
            "draft": "",
            "approved": False,
            "feedback": "",
            "final": "",
        },
        config,
    )

    # ─── PHASE 2: Human provides feedback and resume ───
    current_state = app.get_state(config)
    print(f"   Next node(s): {current_state.next}")

    feedback_text = (
        "Make it more concise and add specific mention of the company culture"
    )

    app.update_state(
        config, {"approved": False, "feedback": feedback_text}  # Request changes
    )

    # # ─── PHASE 3: Continue execution
    final_result = app.invoke(None, config)

    # ─── RESULT ───
    print(f"   Final result ({len(final_result['final'].split())} words):")
    print(f"   {final_result['final'][:200]}...")

class ReviewState(TypedDict):
    document: str
    review_comments: Annotated[list[str], operator.add]
    revision_count: int
    status: str


def demo_iterative_review():

    def submit_for_review(state: ReviewState) -> dict:
        return {"status": "pending_review"}
    
    def human_review(state: ReviewState) -> dict:
        return {}

    def apply_feedback(state: ReviewState) -> dict:
        if not state["review_comments"]:
            print(f"   No comments to apply. Passing through.")
            return state

        feedback = state["review_comments"][-1]

        response = llm.invoke(
            f"Revise this document based on feedback:\n\n"
            f"Document: {state['document']}\n\n"
            f"Feedback: {feedback}"
        )

        return {
            "document": response.content,
            "revision_count": state["revision_count"] + 1,
            "status": "revised",
        }

    def route_after_review(state: ReviewState) -> Literal["apply", "done"]:
        if state["status"] == "approved":
            return "done"
        return "apply"

    def finalize(state: ReviewState) -> dict:
        return {"status": "finalized"}

    graph = StateGraph(ReviewState)

    graph.add_node("submit", submit_for_review)
    graph.add_node("human", human_review)
    graph.add_node("apply", apply_feedback)
    graph.add_node("done", finalize)

    graph.add_edge(START, "submit")
    graph.add_edge("submit", "human")
    graph.add_conditional_edges(
        "human", route_after_review, {"apply": "apply", "done": "done"}
    )
    graph.add_edge("apply", "submit")  # Loop for more reviews
    graph.add_edge("done", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory, interrupt_before=["human"])

    config = {"configurable": {"thread_id": "review-1"}}

    # ─── ROUND 0: Initial submission ───
    result = app.invoke(
        {
            "document": "AI is technology that helps computers think.",
            "review_comments": [],
            "revision_count": 0,
            "status": "",
        },
        config,
    )

    print(f"   Document ready for review: \"{result['document']}\"")
    print(f"   Revisions so far: {result['revision_count']}")

    current_state = app.get_state(config)
    print(f"   Next node: {current_state.next}")

    # ─── ROUND 1: Reviewer wants changes ───
    feedback_1 = "Add more technical depth and examples"
    print(f'   Reviewer says: "{feedback_1}"')

    app.update_state(
        config, {"review_comments": [feedback_1], "status": "needs_revision"}
    )
    result = app.invoke(None, config)

    print(f"   Revised document: {result['document'][:150]}...")
    print(f"   Revisions so far: {result['revision_count']}")

    current_state = app.get_state(config)
    print(f"   Next node: {current_state.next}")

    # ─── ROUND 2: Reviewer wants more changes ───
    feedback_2 = "Good improvement! Now add a concrete example of neural networks"

    app.update_state(
        config, {"review_comments": [feedback_2], "status": "needs_revision"}
    )

    result = app.invoke(None, config)
    print(f"   Revised document: {result['document'][:150]}...")
    print(f"   Revisions so far: {result['revision_count']}")

    # ─── ROUND 3: Reviewer approves ───
    app.update_state(config, {"status": "approved"})
    final = app.invoke(None, config)

    # ─── FINAL SUMMARY ───
    print(f"   Final status: {final['status']}")
    print(f"   Total revisions: {final['revision_count']}")
    print(f"   Final document: {final['document'][:200]}...")

if __name__ == "__main__":
    #demo_interrupt_for_approval()
    demo_iterative_review()