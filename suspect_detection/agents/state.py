from typing import TypedDict, Annotated, Literal, Union, Optional
import operator


def merge_findings(existing: list, new: Union[list, dict]) -> list:
    """Custom reducer to merge findings, handling both list and single dict inputs."""
    if isinstance(new, dict):
        new = [new]
    if not new:
        return existing

    # Use signal as dedup key
    seen = {f.get("signal", str(i)): f for i, f in enumerate(existing)}

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    for finding in new:
        key = finding.get("signal", "")
        if key in seen:
            # Keep higher severity version
            if severity_rank.get(finding.get("severity", "low"), 4) < severity_rank.get(
                seen[key].get("severity", "low"), 4
            ):
                seen[key] = finding
        else:
            seen[key] = finding

    return list(seen.values())


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph."""

    # User input
    user_message: str

    # Patient context
    patient_id: Optional[str]

    # Documents
    documents: list

    # Extracted clinical data
    medications: list
    labs: list
    conditions: list
    prior_year_conditions: list
    symptoms: list

    # Detection results
    findings: Annotated[list, merge_findings]
    completed_strategies: Annotated[list, operator.add]

    # Validation
    validated_findings: list
    findings_to_refine: list
    refinement_attempts: int

    # Control flow
    next_step: str
    current_strategy: Optional[str]
    info_request: bool
    original_query: Optional[str]

    # Error handling
    error: Optional[str]

    # Output
    response: str


def create_initial_state(user_message: str) -> AgentState:
    """Create a properly initialized state for a new conversation turn."""
    return AgentState(
        user_message=user_message,
        patient_id=None,
        documents=[],
        medications=[],
        labs=[],
        conditions=[],
        prior_year_conditions=[],
        symptoms=[],
        findings=[],
        completed_strategies=[],
        validated_findings=[],
        findings_to_refine=[],
        refinement_attempts=0,
        next_step="",
        current_strategy=None,
        info_request=False,
        original_query=None,
        error=None,
        response="",
    )
