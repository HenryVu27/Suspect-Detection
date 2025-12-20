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
    return state.get("next_step", "direct_reply")

def route_from_load_documents(state: AgentState) -> Literal["extraction", "direct_reply"]:
    if state.get("error") or not state.get("documents"):
        return "direct_reply"
    return "extraction"

def route_from_extraction(state: AgentState) -> Literal["supervisor", "answer_query"]:
    if state.get("info_request"):
        return "answer_query"
    return "supervisor"

def route_from_supervisor(state: AgentState) -> Literal["cross_reference", "dropoff", "symptom_cluster", "contradiction", "aggregate"]:
    return state.get("next_step", "aggregate")

def route_from_validation(state: AgentState) -> Literal["refine", "report"]:
    findings_to_refine = state.get("findings_to_refine", [])
    refinement_attempts = state.get("refinement_attempts", 0)
    if findings_to_refine and refinement_attempts < 2:
        return "refine"
    return "report"

def build_graph() -> StateGraph: