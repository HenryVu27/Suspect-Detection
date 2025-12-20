import logging
from difflib import SequenceMatcher

from agents.state import AgentState
from agents.models import INTENT_SCHEMA
from agents.gemini_client import get_gemini_client
from config import GEMINI_FLASH_MODEL, PATIENT_DATA_PATH
from retrieval.search import get_search_index
from retrieval.loader import DocumentLoader

logger = logging.getLogger(__name__)


INTENT_PROMPT = """You are a clinical suspect detection assistant. Classify the user's intent.

**Available Intents:**

1. **analyze_patient**: User explicitly wants to RUN ANALYSIS / DETECT ISSUES for a patient
   - Keywords: "analyze", "check for issues", "detect", "find gaps", "run detection"
   - Examples: "Analyze CVD-2025-001", "Run detection on patient CVD-2025-001"
   - NOT for general information requests about a patient

2. **patient_info_request**: User wants INFORMATION about a patient (NOT a new analysis)
   - User asks about specific aspects: medications, labs, conditions, history, symptoms
   - User asks what to focus on, prioritize, or summarize
   - Examples: "Tell me about CVD-2025-001's medications", "What are the lab results for CVD-2025-001?",
     "What should I focus on for patient CVD-2025-001?", "Summarize CVD-2025-001's conditions"
   - This is different from analyze_patient - it's asking for info, not running detection

3. **list_patients**: User wants to see available patients
   - Examples: "List patients", "What patients are available?", "Show patients"

4. **clarify_patient**: User mentions a patient but ID is incomplete, ambiguous, or invalid
   - Examples: "Tell me about patient CVD", "Analyze patient 2025", "Check CVD-2025"
   - Set needs_clarification=true and include partial_patient_id

5. **followup_question**: User asks a follow-up question WITHOUT mentioning a specific patient ID
   - Refers to "this patient", "the patient", or asks about previously discussed data
   - Examples: "What medications is this patient on?", "What do the lab results suggest?",
     "Are there any gaps?", "Tell me more about the findings"

6. **medical_question**: User asks about medical concepts, conditions, medications, or clinical knowledge
   - Examples: "What is type 1 diabetes?", "Explain HbA1c", "What does metformin treat?"

7. **greeting**: Simple greeting, casual conversation, OR asking about system capabilities/help
   - Examples: "Hi", "Hello", "How are you?", "What can you do?", "Help", "How does this work?"

**CRITICAL DISTINCTION:**
- "Analyze patient CVD-2025-001" -> analyze_patient (run detection)
- "Tell me about CVD-2025-001's medications" -> patient_info_request (get info)
- "What should I focus on for CVD-2025-001?" -> patient_info_request (get info)
- "What medications is this patient on?" -> followup_question (no patient ID mentioned)

**Important:**
- Extract patient_id if it matches format XXX-YYYY-NNN
- Use patient_info_request when user wants specific info WITH a patient ID
- Use followup_question when user wants info WITHOUT specifying a patient ID
"""


def orchestrator_node(state: AgentState) -> dict:
    user_message = state.get("user_message", "").strip()
    logger.info(f"Orchestrator processing: {user_message[:50]}...")

    available_patients = _get_available_patients()

    # Intent classification
    try:
        client = get_gemini_client()
        result = client.generate_structured(
            prompt=f"User message: {user_message}",
            response_schema=INTENT_SCHEMA,
            model=GEMINI_FLASH_MODEL,
            system_instruction=INTENT_PROMPT,
        )

        intent = result.get("intent", "greeting")
        patient_id = result.get("patient_id")
        partial_patient_id = result.get("partial_patient_id", "")
        needs_clarification = result.get("needs_clarification", False)

        logger.info(f"Intent: {intent}, patient_id: {patient_id}, partial: {partial_patient_id}")

        # Handle intents
        if intent == "analyze_patient" and patient_id:
            patient_id, error = _validate_patient_exists(patient_id, available_patients)
            if error:
                suggestion = _find_similar_patient(patient_id, available_patients)
                if suggestion:
                    error["response"] += f"\n\nTry: `Analyze patient {suggestion}`"
                return error
            return {
                "next_step": "analyze",
                "patient_id": patient_id,
                "original_query": user_message,
            }

        elif intent == "analyze_patient" and not patient_id:
            # User wants to analyze but didn't specify which patient
            return _handle_patient_clarification("", available_patients)

        elif intent == "clarify_patient" or needs_clarification:
            return _handle_patient_clarification(partial_patient_id or patient_id, available_patients)

        elif intent == "list_patients":
            return {"next_step": "list_patients"}

        elif intent == "patient_info_request" and patient_id:
            patient_id, error = _validate_patient_exists(patient_id, available_patients)
            if error:
                return error

            # Check existing data
            existing_patient_id = state.get("patient_id")
            has_data = state.get("medications") or state.get("labs") or state.get("conditions")

            if existing_patient_id == patient_id and has_data:
                # Route to answer_query
                return {
                    "next_step": "answer_query",
                    "patient_id": patient_id,
                    "original_query": user_message,
                }
            else:
                # Retrieve data first
                return {
                    "next_step": "retrieve_info",
                    "patient_id": patient_id,
                    "original_query": user_message,
                    "info_request": True,  # Flag to skip detection
                }

        elif intent == "followup_question":
            # Route to answer_query
            return {
                "next_step": "answer_query",
                "patient_id": state.get("patient_id"),
                "original_query": user_message,
            }

        elif intent == "medical_question":
            # Route to dedicated medical QA node
            return {"next_step": "medical_qa"}

        elif intent == "greeting":
            # Handles greetings, help requests, and system capability questions
            return {"next_step": "general_response", "response_type": "greeting"}

        else:
            # Fallback - route to general response node
            return {"next_step": "general_response", "response_type": "fallback"}

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            "next_step": "general_response",
            "response_type": "error",
            "error": str(e),
        }


def _get_available_patients() -> list[str]:
    try:
        index = get_search_index()
        return index.list_patients()
    except Exception as e:
        logger.warning(f"Could not load patient list: {e}")
        return []


def _validate_patient_exists(patient_id: str, available_patients: list[str]) -> tuple[str, dict | None]:
    """Validate patient ID exists and return (upper_id, error_response).

    Returns:
        (patient_id_upper, None) if valid
        (patient_id_upper, error_dict) if invalid
    """
    patient_id = patient_id.upper()
    if not available_patients or patient_id in available_patients:
        return patient_id, None

    suggestion = _find_similar_patient(patient_id, available_patients)
    if suggestion:
        return patient_id, {
            "next_step": "general_response",
            "response_type": "patient_not_found",
            "response": f"Patient '{patient_id}' not found. Did you mean **{suggestion}**?",
        }
    return patient_id, {
        "next_step": "general_response",
        "response_type": "patient_not_found",
        "response": f"Patient '{patient_id}' not found.\n\nAvailable patients:\n"
                    + "\n".join(f"- {p}" for p in available_patients[:5]),
    }


def _find_similar_patient(query: str, available: list[str], threshold: float = 0.6) -> str | None:
    if not available:
        return None

    query_lower = query.lower()
    best_match = None
    best_score = 0

    for patient_id in available:
        # Substring match
        if query_lower in patient_id.lower():
            return patient_id

        score = SequenceMatcher(None, query_lower, patient_id.lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = patient_id

    return best_match


def _handle_patient_clarification(partial_id: str, available_patients: list[str]) -> dict:
    if not partial_id:
        if available_patients:
            patient_list = "\n".join(f"- {p}" for p in available_patients[:10])
            return {
                "next_step": "general_response",
                "response_type": "patient_clarification",
                "response": f"Which patient would you like to analyze?\n\n**Available patients:**\n{patient_list}",
            }
        return {
            "next_step": "general_response",
            "response_type": "patient_clarification",
            "response": "Please specify a patient ID in the format XXX-YYYY-NNN (e.g., CVD-2025-001).",
        }

    # Find matches
    matches = []
    partial_lower = partial_id.lower()

    for patient_id in available_patients:
        if partial_lower in patient_id.lower():
            matches.append(patient_id)

    if len(matches) == 1:
        return {
            "next_step": "general_response",
            "response_type": "patient_clarification",
            "response": f"Did you mean **{matches[0]}**?\n\nTry: `Analyze patient {matches[0]}`",
        }
    elif len(matches) > 1:
        match_list = "\n".join(f"- {p}" for p in matches[:5])
        return {
            "next_step": "general_response",
            "response_type": "patient_clarification",
            "response": f"Multiple patients match '{partial_id}':\n{match_list}\n\nPlease specify the full patient ID.",
        }
    else:
        if available_patients:
            patient_list = "\n".join(f"- {p}" for p in available_patients[:5])
            return {
                "next_step": "general_response",
                "response_type": "patient_clarification",
                "response": f"No patient found matching '{partial_id}'.\n\n**Available patients:**\n{patient_list}",
            }
        return {
            "next_step": "general_response",
            "response_type": "patient_clarification",
            "response": f"Patient '{partial_id}' not found. Please check the ID format (XXX-YYYY-NNN).",
        }


def list_patients_node(_state: AgentState) -> dict:
    try:
        # Search index
        index = get_search_index()
        patients = index.list_patients()

        # Fallback
        if not patients:
            loader = DocumentLoader(PATIENT_DATA_PATH)
            patients = loader.list_patients()

        if patients:
            patient_list = "\n".join(f"- {p}" for p in patients)
            response = f"**Available Patients ({len(patients)}):**\n{patient_list}"
        else:
            response = "No patients found in the system."

        return {"response": response, "next_step": "end"}

    except Exception as e:
        logger.error(f"Failed to list patients: {e}")
        return {
            "response": f"Error listing patients: {str(e)}",
            "error": str(e),
            "next_step": "end",
        }
