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
from agents.nodes.report import report_node, quick_report_node
from agents.nodes.answer_query import answer_query_node

logger = logging.getLogger(__name__)

