import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import AgentState, create_initial_state

# Import all node functions
from agents.nodes.orchestrator import orchestrator_node, list_patients_node
from agents.nodes.documents import load_documents_node
from agents.nodes.extraction import extraction_node
from agents.nodes.detection import (
    cross_reference_node,
    dropoff_node,
    symptom_cluster_node,
    contradiction_node,
    aggregate_findings_node,
)
from agents.nodes.supervisor import supervisor_node
from agents.nodes.validation import self_reflect_node, refine_node
from agents.nodes.report import report_node, quick_report_node
from agents.nodes.answer_query import answer_query_node

logger = logging.getLogger(__name__)


def route_from_orchestrator(state: AgentState) -> Literal["list_patients", "analyze", "retrieve_info", "direct_reply"]:
    # Route based on orchestrator's intent classification
    next_step = state.get("next_step", "direct_reply")
    valid_steps = {"list_patients", "analyze", "retrieve_info"}
    return next_step if next_step in valid_steps else "direct_reply"


def route_from_load_documents(state: AgentState) -> Literal["extraction", "direct_reply"]:
    # Route based on document retrieval result
    if state.get("error") or not state.get("documents"):
        return "direct_reply"
    return "extraction"


def route_from_extraction(state: AgentState) -> Literal["supervisor", "answer_query"]:
    # Route based on whether this is an info request or full analysis
    if state.get("info_request"):
        return "answer_query"
    return "supervisor"


def route_from_supervisor(state: AgentState) -> Literal[
    "cross_reference", "dropoff", "symptom_cluster", "contradiction", "aggregate"
]:
    # Route based on supervisor's decision
    next_step = state.get("next_step", "aggregate")
    valid_steps = {"cross_reference", "dropoff", "symptom_cluster", "contradiction"}
    return next_step if next_step in valid_steps else "aggregate"


def route_from_validation(state: AgentState) -> Literal["refine", "report"]:
    # Route based on validation results
    findings_to_refine = state.get("findings_to_refine", [])
    refinement_attempts = state.get("refinement_attempts", 0)

    if findings_to_refine and refinement_attempts < 2:
        return "refine"
    return "report"


def build_graph() -> StateGraph:
    """Build the complete LangGraph workflow.

    Returns:
        Uncompiled StateGraph
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("list_patients", list_patients_node)
    workflow.add_node("direct_reply", quick_report_node)

    # Document retrieval (uses hybrid search)
    workflow.add_node("load_documents", load_documents_node)

    # Extraction
    workflow.add_node("extraction", extraction_node)

    # Answer query (for info requests - bypasses detection)
    workflow.add_node("answer_query", answer_query_node)

    # Detection (Supervisor pattern)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("cross_reference", cross_reference_node)
    workflow.add_node("dropoff", dropoff_node)
    workflow.add_node("symptom_cluster", symptom_cluster_node)
    workflow.add_node("contradiction", contradiction_node)
    workflow.add_node("aggregate", aggregate_findings_node)

    # Validation (Self-RAG)
    workflow.add_node("self_reflect", self_reflect_node)
    workflow.add_node("refine", refine_node)

    # Report
    workflow.add_node("report", report_node)

    # Edges
    workflow.add_edge(START, "orchestrator")

    # Orchestrator routing
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "list_patients": "list_patients",
            "analyze": "load_documents",
            "retrieve_info": "load_documents",  # Same path, but info_request flag is set
            "direct_reply": "direct_reply",
        },
    )

    # Terminal nodes for simple paths
    workflow.add_edge("list_patients", END)
    workflow.add_edge("direct_reply", END)

    # Document retrieval with error handling
    workflow.add_conditional_edges(
        "load_documents",
        route_from_load_documents,
        {
            "extraction": "extraction",
            "direct_reply": "direct_reply",
        },
    )

    # Extraction -> either Supervisor (for analysis) or answer_query (for info requests)
    workflow.add_conditional_edges(
        "extraction",
        route_from_extraction,
        {
            "supervisor": "supervisor",
            "answer_query": "answer_query",
        },
    )

    # Answer query terminal
    workflow.add_edge("answer_query", END)

    # Supervisor routing to detection agents
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "cross_reference": "cross_reference",
            "dropoff": "dropoff",
            "symptom_cluster": "symptom_cluster",
            "contradiction": "contradiction",
            "aggregate": "aggregate",
        },
    )

    # Detection agents return to supervisor
    workflow.add_edge("cross_reference", "supervisor")
    workflow.add_edge("dropoff", "supervisor")
    workflow.add_edge("symptom_cluster", "supervisor")
    workflow.add_edge("contradiction", "supervisor")

    # Aggregate -> Self-reflect
    workflow.add_edge("aggregate", "self_reflect")

    # Validation with optional refinement loop
    workflow.add_conditional_edges(
        "self_reflect",
        route_from_validation,
        {
            "refine": "refine",
            "report": "report",
        },
    )
    workflow.add_edge("refine", "self_reflect")  # Loop back to re-validate refined findings

    # Report -> End
    workflow.add_edge("report", END)

    logger.info("Graph built with all nodes and edges")
    return workflow


def create_graph(checkpointer=None):
    """Create compiled graph with checkpointing.

    Args:
        checkpointer: Pre-configured checkpointer (defaults to MemorySaver)

    Returns:
        Compiled graph ready for invocation
    """
    workflow = build_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# Keys to preserve from previous state for context continuity
_CONTEXT_KEYS = [
    "patient_id", "documents", "medications", "labs", "conditions",
    "prior_year_conditions", "symptoms", "validated_findings", "findings"
]


def _preserve_context(initial_state: dict, previous_state: dict) -> None:
    """Preserve relevant context from previous state for follow-up questions.

    Modifies initial_state in place.
    """
    if not previous_state:
        return

    for key in _CONTEXT_KEYS:
        if previous_state.get(key):
            initial_state[key] = previous_state[key]


async def run_analysis(
    user_message: str,
    thread_id: str = None,
    graph=None,
    previous_state: dict = None,
) -> dict:
    """Run analysis with the LangGraph workflow.

    Args:
        user_message: User's message/query
        thread_id: Optional thread ID for checkpointing
        graph: Optional pre-compiled graph
        previous_state: Optional previous state to preserve context

    Returns:
        Final state dict with response
    """
    if graph is None:
        graph = create_graph()

    # Start with fresh state, preserving context from previous if available
    initial_state = create_initial_state(user_message)
    _preserve_context(initial_state, previous_state)

    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    return await graph.ainvoke(initial_state, config=config)


def run_analysis_sync(
    user_message: str,
    thread_id: str = None,
    graph=None,
    previous_state: dict = None,
) -> dict:
    """Synchronous version of run_analysis.

    Args:
        user_message: User's message/query
        thread_id: Optional thread ID for checkpointing
        graph: Optional pre-compiled graph
        previous_state: Optional previous state to preserve context

    Returns:
        Final state dict with response
    """
    if graph is None:
        graph = create_graph()

    # Start with fresh state, preserving context from previous if available
    initial_state = create_initial_state(user_message)
    _preserve_context(initial_state, previous_state)

    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    return graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        print("Testing LangGraph workflow...")

        # Test list patients
        result = await run_analysis("List available patients")
        print(f"\n--- List Patients ---\n{result.get('response', 'No response')}")

        # Test analyze patient (if patient exists)
        result = await run_analysis("Analyze patient CVD-2025-001")
        print(f"\n--- Analyze Patient ---\n{result.get('response', 'No response')}")

    asyncio.run(main())
