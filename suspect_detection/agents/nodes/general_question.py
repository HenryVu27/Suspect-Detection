import logging

from agents.state import AgentState
from agents.gemini_client import get_gemini_client
from config import GEMINI_FLASH_MODEL

logger = logging.getLogger(__name__)

GREETING_RESPONSE = """Hello! I'm a **Clinical Suspect Detection System**.

I can help you with:

1. **Analyze a patient** - Detect gaps and suspect conditions
   - Example: "Analyze patient CVD-2025-001"

2. **List available patients**
   - Example: "List patients"

3. **Answer medical questions**
   - Example: "What is HbA1c?"

**What I detect:**
- Medications without documented diagnoses
- Abnormal labs without corresponding conditions
- Chronic conditions missing from current records
- Symptom patterns suggesting undiagnosed conditions
- Contradictions in clinical documentation

Try: "Analyze patient CVD-2025-001" to get started."""

ERROR_RESPONSE = """I encountered an issue processing your request.

Try:
- `List patients` - see available patients
- `Analyze patient CVD-2025-001` - analyze a specific patient
- `What is diabetes?` - ask a medical question"""

FALLBACK_SYSTEM_INSTRUCTION = (
    "You are a clinical assistant. Answer helpfully and briefly. "
    "If the user seems to want patient analysis, guide them to use "
    "'Analyze patient <ID>' format."
)


def general_question_node(state: AgentState) -> dict:
    """Handle general responses: greetings, help, errors, clarifications, and fallback."""
    response_type = state.get("response_type", "fallback")
    user_message = state.get("user_message", "")

    logger.info(f"General response for type: {response_type}")

    # If orchestrator already set a response (e.g., patient_not_found, patient_clarification)
    # just pass it through
    existing_response = state.get("response")
    if existing_response and response_type in ("patient_not_found", "patient_clarification"):
        return {"response": existing_response, "next_step": "end"}

    # Handle known static response types
    if response_type == "greeting":
        return {"response": GREETING_RESPONSE, "next_step": "end"}

    if response_type == "error":
        return {"response": ERROR_RESPONSE, "next_step": "end"}

    # Fallback: use LLM for unknown intents
    try:
        client = get_gemini_client()
        response = client.generate(
            prompt=user_message,
            model=GEMINI_FLASH_MODEL,
            system_instruction=FALLBACK_SYSTEM_INSTRUCTION,
        )
        return {"response": response, "next_step": "end"}

    except Exception as e:
        logger.error(f"General response failed: {e}")
        return {
            "response": "I'm not sure how to help with that. Try 'List patients' or 'Analyze patient CVD-2025-001'.",
            "error": str(e),
            "next_step": "end",
        }

        
