from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gpt-4o-mini", temperature=0.0)

# Basic Routing
class RouterState(TypedDict):
    query: str
    query_type: str
    response: str

def demo_basic_routing():
    def classify_query(state: RouterState) -> dict:
        response = llm.invoke(
            f"Classify this query as 'question', 'command', or 'statement'. "
            f"Reply with just the word.\n\n{state['query']}"
        )
        return {"query_type": response.content.lower().strip()}

    def handle_question(state: RouterState) -> dict:
        response = llm.invoke(f"Answer this question: {state['query']}")
        return {"response": f"[Answer] {response.content}"}

    def handle_command(state: RouterState) -> dict:
        return {"response": f"[Executing] I'll help you with: {state['query']}"}

    def handle_statement(state: RouterState) -> dict:
        return {"response": f"[Acknowledged] Thanks for sharing: {state['query']}"}

    def route_by_type(
        state: RouterState,
    ) -> Literal["question", "command", "statement"]:
        qt = state["query_type"]
        if "question" in qt:
            return "question"
        elif "command" in qt:
            return "command"
        else:
            return "statement"

    graph = StateGraph(RouterState)

    graph.add_node("classify", classify_query)
    graph.add_node("handle_question", handle_question)
    graph.add_node("handle_command", handle_command)
    graph.add_node("handle_statement", handle_statement)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",  # source node
        route_by_type,  # function that determines which edge to take based on the state
        {
            "question": "handle_question",
            "command": "handle_command",
            "statement": "handle_statement",
        },
    )

    graph.add_edge("handle_question", END)
    graph.add_edge("handle_command", END)
    graph.add_edge("handle_statement", END)

    app = graph.compile()

    # Example usage
    queries = [
        "What is the capital of France?",
        "Send an email to John",
        "I love programming",
    ]

    for query in queries:
        result = app.invoke({"query": query})
        print(f"Query: {query}")
        print(f"Type: {result['query_type']}")
        print(f"Response: {result['response']}")
        print("-" * 40)

# Basic Conditional Loop
class QualityState(TypedDict):
    content: str
    quality_score: int
    feedback: str
    final_content: str
    iteration: int


def demo_conditional_loop():

    def evaluate_quality(state: QualityState) -> dict:
        response = llm.invoke(
            f"Rate this content quality from 1-10. Reply with just the number.\n\n"
            f"Content: {state['content']}"
        )
        try:
            score = int(response.content.strip())
        except:
            score = 5
        return {"quality_score": score}

    def improve_content(state: QualityState) -> dict:
        response = llm.invoke(
            f"Improve this content to be more engaging and clear:\n\n{state['content']}"
        )
        return {"content": response.content, "iteration": state["iteration"] + 1}

    def finalize_content(state: QualityState) -> dict:
        return {
            "final_content": state["content"],
            "feedback": f"Approved after {state['iteration']} iterations with score {state['quality_score']}",
        }

    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        if state["quality_score"] >= 7:
            return "finalize"
        elif state["iteration"] >= 3:
            return "finalize"  # Max iterations
        else:
            return "improve"

    graph = StateGraph(QualityState)

    graph.add_node("evaluate", evaluate_quality)
    graph.add_node("improve", improve_content)
    graph.add_node("finalize", finalize_content)

    graph.add_edge(START, "evaluate")

    graph.add_conditional_edges(
        "evaluate", should_continue, {"improve": "improve", "finalize": "finalize"}
    )

    graph.add_edge("improve", "evaluate")  # Loop back!
    graph.add_edge("finalize", END)

    app = graph.compile()

    result = app.invoke(
        {
            "content": "AI is cool",
            "quality_score": 0,
            "feedback": "",
            "final_content": "",
            "iteration": 0,
        }
    )

    print(f"Original: AI is cool")
    print(f"Final: {result['final_content'][:200]}...")
    print(f"Feedback: {result['feedback']}")

# Basic Multi-Path Routing
def demo_multi_path_routing():
    class TaskState(TypedDict):
        task: str
        urgency: str
        complexity: str
        handler: str
        result: str

    def analyze_task(state: TaskState) -> dict:
        # Analyze urgency
        urgency_response = llm.invoke(
            f"Is this task urgent? Reply 'urgent' or 'normal'.\nTask: {state['task']}"
        )

        # Analyze complexity
        complexity_response = llm.invoke(
            f"Is this task complex? Reply 'complex' or 'simple'.\nTask: {state['task']}"
        )

        return {
            "urgency": urgency_response.content.lower().strip(),
            "complexity": complexity_response.content.lower().strip(),
        }

    def urgent_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Senior Team",
            "result": "Escalated to senior team for immediate action",
        }

    def urgent_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Quick Response",
            "result": "Handled immediately by available agent",
        }

    def normal_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Specialist",
            "result": "Assigned to specialist for thorough handling",
        }

    def normal_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Standard",
            "result": "Added to standard queue",
        }

    def route_task(state: TaskState) -> str:
        is_urgent = "urgent" in state["urgency"]
        is_complex = "complex" in state["complexity"]

        if is_urgent and is_complex:
            return "urgent_complex"
        elif is_urgent:
            return "urgent_simple"
        elif is_complex:
            return "normal_complex"
        else:
            return "normal_simple"

    graph = StateGraph(TaskState)

    graph.add_node("analyze", analyze_task)
    graph.add_node("urgent_complex", urgent_complex_handler)
    graph.add_node("urgent_simple", urgent_simple_handler)
    graph.add_node("normal_complex", normal_complex_handler)
    graph.add_node("normal_simple", normal_simple_handler)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_task,
        {
            "urgent_complex": "urgent_complex",
            "urgent_simple": "urgent_simple",
            "normal_complex": "normal_complex",
            "normal_simple": "normal_simple",
        },
    )

    for node in ["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
        graph.add_edge(node, END)

    app = graph.compile()

    tasks = [
        "Server is down! Need immediate fix!",
        "Update the documentation for the API",
        "Redesign the entire database schema",
        "Fix the typo on the homepage",
    ]

    for task in tasks:
        result = app.invoke({"task": task})
        print(f"Task: {task}")
        print(f"Urgency: {result['urgency']} | Complexity: {result['complexity']}")
        print(f"Handler: {result['handler']}")
        print(f"Result: {result['result']}")
        print("-" * 40)

# iterative research demo
class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str


def demo_iterative_research():
    """Iterative research that goes deeper based on findings."""

    def research(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(f"📚 [RESEARCH] Depth {state['iteration'] + 1}/{state['max_depth']}")

        if state["iteration"] == 0:
            query = f"Give me 3 key facts about: {state['topic']}"
            print(f"   Starting fresh on: {state['topic']}")
        else:
            question = state["questions"][-1] if state["questions"] else "elaborate"
            query = f"Based on these findings:\n{state['findings'][-1]}\n\nGo deeper: {question}"
            print(f"   Following up on: {question}")

        response = llm.invoke(query)
        print(f"   ✅ Found {len(response.content.splitlines())} lines of findings")
        print(f"   Preview: {response.content[:120]}...")
        return {"findings": [response.content]}

    def generate_questions(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(f"🤔 [QUESTIONING] Analyzing latest findings...")

        response = llm.invoke(
            f"Based on this finding:\n{state['findings'][-1]}\n\n"
            "What's one deeper question to explore? Reply with just the question."
        )

        print(f"   Next question: {response.content.strip()}")

        return {"questions": [response.content], "iteration": state["iteration"] + 1}

    def synthesize(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(
            f"🧬 [SYNTHESIZE] Combining {len(state['findings'])} rounds of findings..."
        )

        all_findings = "\n\n".join(state["findings"])
        response = llm.invoke(
            f"Synthesize these findings into a coherent summary:\n\n{all_findings}"
        )

        print(f"   ✅ Summary generated ({len(response.content.split())} words)")
        return {"summary": response.content}

    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        if state["iteration"] >= state["max_depth"]:
            print(
                f"\n🏁 [ROUTER] Max depth reached ({state['iteration']}/{state['max_depth']}) → synthesizing"
            )
            return "synthesize"
        print(
            f"\n🔄 [ROUTER] Depth {state['iteration']}/{state['max_depth']} → going deeper"
        )
        return "research"

    graph = StateGraph(ResearchState)

    graph.add_node("research", research)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "research")
    graph.add_edge("research", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        should_continue,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile()

    print("=" * 50)
    print("🔬 ITERATIVE RESEARCH WORKFLOW")
    print("=" * 50)

    result = app.invoke(
        {
            "topic": "quantum computing applications",
            "findings": [],
            "questions": [],
            "iteration": 0,
            "max_depth": 2,
            "summary": "",
        }
    )

    print(f"\n{'=' * 50}")
    print(f"📊 RESEARCH COMPLETE")
    print(f"   Topic: {result['topic']}")
    print(f"   Depth reached: {result['iteration']}")
    print(f"   Findings collected: {len(result['findings'])}")
    print(f"   Questions explored: {len(result['questions'])}")
    print(f"\n📝 Final Summary:\n{result['summary']}")

if __name__ == "__main__":
    #demo_basic_routing()
    #demo_conditional_loop()
    #demo_multi_path_routing()
    demo_iterative_research()