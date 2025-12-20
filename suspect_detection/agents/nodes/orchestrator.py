import logging
from difflib import SequenceMatcher

from agents.state import AgentState
from agents.models import INTENT_SCHEMA
from agents.gemini_client import get_gemini_client
from agents.utils import build_patient_context
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

7. **system_help**: User asks about what this system can do or how to use it
   - Examples: "What can you do?", "Help", "How does this work?"

8. **greeting**: Simple greeting or casual conversation
   - Examples: "Hi", "Hello", "How are you?"

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

MEDICAL_QA_PROMPT = """You are a knowledgeable clinical assistant helping healthcare professionals.

Provide accurate, concise medical information. Include:
- Clear definition/explanation
- Clinical relevance
- Key points a clinician should know

Keep responses focused and professional. Use medical terminology appropriately.
If the question is outside your knowledge or requires patient-specific advice, say so.
"""

SYSTEM_HELP_RESPONSE = """**Clinical Suspect Detection System**

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

Try: "Analyze patient CVD-2025-001" to get started.
"""


def orchestrator_node(state: AgentState) -> dict:
    user_message = state.get("user_message", "").strip()
    if not user_message:
        return {
            "next_step": "direct_reply",
            "response": "Please enter a message. Try 'List patients' or 'Analyze patient CVD-2025-001'.",
        }

    logger.info(f"Orchestrator processing: {user_message[:50]}...")

    # Get available patients
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
                # Add action hint for analyze intent
                suggestion = _find_similar_patient(patient_id, available_patients)
                if suggestion:
                    error["response"] += f"\n\nTry: `Analyze patient {suggestion}`"
                return error
            return {
                "next_step": "analyze",
                "patient_id": patient_id,
                "original_query": user_message,
            }

        elif intent == "clarify_patient" or needs_clarification:
            # User mentioned patient but ID is incomplete
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
                # Answer directly
                context = build_patient_context(state)
                response = client.generate(
                    prompt=f"Patient data:\n{context}\n\nUser question: {user_message}",
                    model=GEMINI_FLASH_MODEL,
                    system_instruction="You are a clinical assistant. Answer the user's specific question based on the patient data provided. "
                                       "Be concise and directly address what they asked. Reference specific data points.",
                )
                return {
                    "next_step": "direct_reply",
                    "response": response,
                    "patient_id": patient_id,
                }
            else:
                # Retrieve data first
                return {
                    "next_step": "retrieve_info",
                    "patient_id": patient_id,
                    "original_query": user_message,
                    "info_request": True,  # Flag to skip detection strategies
                }

        elif intent == "followup_question":
            # Follow-up question
            existing_patient_id = state.get("patient_id")

            if not existing_patient_id:
                return {
                    "next_step": "direct_reply",
                    "response": "No patient has been analyzed yet in this session.\n\n"
                               "Please analyze a patient first:\n"
                               "- `Analyze patient CVD-2025-001`\n"
                               "- `List patients` to see available patients",
                }

            # Build response
            context = build_patient_context(state)
            response = client.generate(
                prompt=f"Patient context:\n{context}\n\nUser question: {user_message}",
                model=GEMINI_FLASH_MODEL,
                system_instruction="You are a clinical assistant. Answer the question based on the patient context provided. "
                                   "Be specific and reference the actual patient data. If you need more information, say so.",
            )
            return {
                "next_step": "direct_reply",
                "response": response,
                "patient_id": existing_patient_id,  # Preserve context
            }

        elif intent == "medical_question":
            # Medical question
            response = client.generate(
                prompt=f"Question: {user_message}",
                model=GEMINI_FLASH_MODEL,
                system_instruction=MEDICAL_QA_PROMPT,
            )
            return {
                "next_step": "direct_reply",
                "response": response,
            }

        elif intent == "system_help":
            return {
                "next_step": "direct_reply",
                "response": SYSTEM_HELP_RESPONSE,
            }

        elif intent == "greeting":
            return {
                "next_step": "direct_reply",
                "response": "Hello! I'm a clinical suspect detection assistant. "
                           "I can analyze patient records for gaps and potential issues.\n\n"
                           "Try: `Analyze patient CVD-2025-001`",
            }

        else:
            # Fallback
            response = client.generate(
                prompt=user_message,
                model=GEMINI_FLASH_MODEL,
                system_instruction="You are a clinical assistant. Answer helpfully and briefly. "
                                   "If the user seems to want patient analysis, guide them to use "
                                   "'Analyze patient <ID>' format.",
            )
            return {
                "next_step": "direct_reply",
                "response": response,
            }

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            "next_step": "direct_reply",
            "response": "I encountered an issue processing your request.\n\n"
                       "Try:\n- `List patients` - see available patients\n"
                       "- `Analyze patient CVD-2025-001` - analyze a specific patient\n"
                       "- `What is diabetes?` - ask a medical question",
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
            "next_step": "direct_reply",
            "response": f"Patient '{patient_id}' not found. Did you mean **{suggestion}**?",
        }
    return patient_id, {
        "next_step": "direct_reply",
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
                "next_step": "direct_reply",
                "response": f"Which patient would you like to analyze?\n\n**Available patients:**\n{patient_list}",
            }
        return {
            "next_step": "direct_reply",
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
            "next_step": "direct_reply",
            "response": f"Did you mean **{matches[0]}**?\n\nTry: `Analyze patient {matches[0]}`",
        }
    elif len(matches) > 1:
        match_list = "\n".join(f"- {p}" for p in matches[:5])
        return {
            "next_step": "direct_reply",
            "response": f"Multiple patients match '{partial_id}':\n{match_list}\n\nPlease specify the full patient ID.",
        }
    else:
        if available_patients:
            patient_list = "\n".join(f"- {p}" for p in available_patients[:5])
            return {
                "next_step": "direct_reply",
                "response": f"No patient found matching '{partial_id}'.\n\n**Available patients:**\n{patient_list}",
            }
        return {
            "next_step": "direct_reply",
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
