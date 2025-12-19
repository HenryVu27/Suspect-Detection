import logging
from typing import Literal, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

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
from agents.nodes.report import report_node
from agents.nodes.answer_query import answer_query_node

logger = logging.getLogger(__name__)

def route_from_orchestrator(state: AgentState) -> Literal["list_patients", "analyze", "retrieve_info", "direct_reply"]:
    # Route based on user intent
    next_step = state.get("next_step", "direct_reply")
    if next_step == "list_patients":
        return ["list_patients"]
    elif next_step == "analyze":
        return "analyze"
    