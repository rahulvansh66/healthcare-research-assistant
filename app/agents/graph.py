from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from app.agents.state import AgentState
from app.config import settings
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.evidence import evidence_agent_node
from app.agents.nodes.responder import generate_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("evidence_agent", evidence_agent_node)
workflow.add_node("responder", generate_node)

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"

workflow.set_entry_point("planner")


# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)


workflow.add_edge("retriever", "evidence_agent")
workflow.add_edge("evidence_agent", "responder")
workflow.add_edge("responder", END)


# --- MEMORY UPGRADE ---
# PostgresSaver persists conversations by 'thread_id' so history (and the
# sessions list) survives backend restarts. from_conn_string() is a
# contextmanager; entered manually (never exited) so the pooled connection
# lives for the process's lifetime, matching this module-level singleton.
_checkpointer_cm = PostgresSaver.from_conn_string(settings.DATABASE_URL)
checkpointer = _checkpointer_cm.__enter__()
checkpointer.setup()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
