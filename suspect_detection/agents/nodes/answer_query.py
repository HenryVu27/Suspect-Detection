import logging

from agents.state import AgentState
from agents.gemini_client import get_gemini_client
from agents.utils import build_patient_context
from config import GEMINI_FLASH_MODEL

logger = logging.getLogger(__name__)


ANSWER_QUERY_PROMPT = """You are a clinical assistant helping healthcare professionals access patient data.

Based on the extracted patient data provided, answer the user's specific question.

Guidelines:
- Be concise and directly address what the user asked
- Reference specific data points from the patient records
- If the user asked about labs, focus on labs (show values, units, flags)
- If the user asked about medications, focus on medications (show name, dose, frequency)
- If the user asked about conditions, focus on conditions
- Format data clearly using markdown
- If the requested data is not available, say so clearly

Do NOT:
- Perform clinical analysis or make diagnoses
- Suggest what conditions the patient might have
- Run detection algorithms or find gaps
- Add unsolicited clinical interpretations

Just present the requested data clearly and concisely.
"""


def answer_query_node(state: AgentState) -> dict:
    patient_id = state.get("patient_id", "Unknown")
    original_query = state.get("original_query", "")

    logger.info(f"Answering info query for patient {patient_id}: {original_query[:50]}...")

    context = build_patient_context(state)

    try:
        client = get_gemini_client()
        response = client.generate(
            prompt=f"Patient data:\n{context}\n\nUser question: {original_query}",
            model=GEMINI_FLASH_MODEL,
            system_instruction=ANSWER_QUERY_PROMPT,
        )

        logger.info(f"Generated info response for patient {patient_id}")

        return {
            "response": response,
            "patient_id": patient_id,
            # Preserve extracted data for follow-up questions
            "next_step": "end",
        }

    except Exception as e:
        logger.error(f"Failed to answer query: {e}")
        return {
            "response": f"I retrieved the patient data but encountered an error generating the response: {str(e)}\n\n"
                       f"Here's the raw data:\n{context}",
            "patient_id": patient_id,
            "error": str(e),
            "next_step": "end",
        }
