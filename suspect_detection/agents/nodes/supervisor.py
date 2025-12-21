import logging

from agents.state import AgentState
from agents.models import SUPERVISOR_DECISION_SCHEMA
from agents.gemini_client import get_gemini_client
from config import GEMINI_FLASH_MODEL

logger = logging.getLogger(__name__)

# Detection strategies
DETECTION_STRATEGIES = ["cross_reference", "dropoff", "symptom_cluster", "contradiction"]

SUPERVISOR_PROMPT = """You are a clinical detection supervisor coordinating specialized agents.

Available detection agents:
1. **cross_reference**: Detects medication/lab-diagnosis gaps
   - Use when: medications or labs are present
   - Finds: medications without diagnoses, abnormal labs without diagnoses

2. **dropoff**: Detects chronic conditions missing from current documentation
   - Use when: prior year conditions exist
   - Finds: chronic conditions documented previously but not currently

3. **symptom_cluster**: Identifies symptom patterns suggesting undiagnosed conditions
   - Use when: symptoms are reported
   - Finds: patterns like sleep apnea, depression, heart failure

4. **contradiction**: Finds conflicting information across documents
   - Use when: conditions have varied statuses or multiple data sources
   - Finds: resolved conditions with active meds, conflicting data

Choose "aggregate" when all relevant strategies have run (based on available data).
Choose "FINISH" to skip detection (e.g., if no useful data).

Consider:
- Which strategies are relevant given the available data
- Don't run strategies that have no relevant input data
- Run cross_reference first if medications/labs exist (most common findings)
"""


def supervisor_node(state: AgentState) -> dict:
    medications = state.get("medications", [])
    labs = state.get("labs", [])
    conditions = state.get("conditions", [])
    prior_conditions = state.get("prior_year_conditions", [])
    symptoms = state.get("symptoms", [])
    completed = state.get("completed_strategies", [])
    current_findings = state.get("findings", [])

    logger.info(f"Supervisor: {len(completed)}/{len(DETECTION_STRATEGIES)} strategies completed")

    # All strategies done -> aggregate
    remaining = [s for s in DETECTION_STRATEGIES if s not in completed]
    if not remaining:
        logger.info("All strategies completed, aggregating")
        return {"next_step": "aggregate"}

    # Find relevant strategies
    relevant_strategies = []

    if (medications or labs) and "cross_reference" not in completed:
        relevant_strategies.append("cross_reference")

    if prior_conditions and "dropoff" not in completed:
        relevant_strategies.append("dropoff")

    if symptoms and "symptom_cluster" not in completed:
        relevant_strategies.append("symptom_cluster")

    if conditions and medications and "contradiction" not in completed:
        relevant_strategies.append("contradiction")

    # No relevant strategies -> aggregate
    if not relevant_strategies:
        logger.info("No relevant strategies remaining, aggregating")
        return {"next_step": "aggregate"}

    # Simple case: single strategy
    if len(relevant_strategies) == 1:
        next_strategy = relevant_strategies[0]
        logger.info(f"Single relevant strategy: {next_strategy}")
        return {
            "next_step": next_strategy,
            "current_strategy": next_strategy,
        }

    # LLM routing for complex cases
    try:
        client = get_gemini_client()
        result = client.generate_structured(
            prompt=f"""Current state:
- Medications: {len(medications)}
- Labs: {len(labs)}
- Conditions: {len(conditions)}
- Prior year conditions: {len(prior_conditions)}
- Symptoms: {len(symptoms)}
- Completed strategies: {completed}
- Current findings: {len(current_findings)}
- Relevant remaining: {relevant_strategies}

Which detection agent should run next?""",
            response_schema=SUPERVISOR_DECISION_SCHEMA,
            model=GEMINI_FLASH_MODEL,
            system_instruction=SUPERVISOR_PROMPT,
        )

        next_agent = result.get("next_agent", "aggregate")
        reasoning = result.get("reasoning", "")

        logger.info(f"Supervisor decided: {next_agent} ({reasoning})")

        if next_agent in ("FINISH", "aggregate"):
            return {"next_step": "aggregate"}
        if next_agent in DETECTION_STRATEGIES and next_agent not in completed:
            return {
                "next_step": next_agent,
                "current_strategy": next_agent,
            }
        else:
            # Fallback
            return {
                "next_step": relevant_strategies[0],
                "current_strategy": relevant_strategies[0],
            }

    except Exception as e:
        logger.error(f"Supervisor LLM call failed: {e}")
        # Fallback
        return {
            "next_step": relevant_strategies[0],
            "current_strategy": relevant_strategies[0],
        }
