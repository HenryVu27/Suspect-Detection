#med
import logging
from re import M

from agents.state import AgentState
from agents.gemini_client import get_gemini_client
from config import GEMINI_FLASH_MODEL

logger = logging.getLogger(__name__)

MEDICAL_QA_PROMPT = """You are a knowledgeable clinical assistant helping healthcare professionals.

Provide accurate, concise medical information. Include:
- Clear definition/explanation
- Clinical relevance
- Key points a clinician should know

Keep responses focused and professional. Use medical terminology appropriately.
If the question is outside your knowledge or requires patient-specific advice, say so.
"""

def medical_qa_node(state: AgentState) -> dict:
    user_message = state.get("user_message", "")
    logger.info(f"Answering medical question: {user_message[:50]}...")

    try:
        client = get_gemini_client()
        response = client.generate(
            prompt=f"Question: {user_message}",
            model=GEMINI_FLASH_MODEL,
            system_instruction=MEDICAL_QA_PROMPT,
        )
        return {"response": response, "next_step": "end"}

    except Exception as e:
        logger.error(f"Failed to answer medical question: {e}")
        return {
            "response": f"I encountered an issue answering your medical question: {str(e)}",
            "error": str(e),    
            "next_step": "end"
        }
        