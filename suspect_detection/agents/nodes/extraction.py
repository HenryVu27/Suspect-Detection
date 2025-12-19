import logging

from agents.state import AgentState
from agents.models import EXTRACTION_SCHEMA
from agents.gemini_client import get_gemini_client
from config import GEMINI_FLASH_MODEL

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract clinical entities from the patient documents.

Be thorough and extract ALL instances of:

1. **Medications**: Drug name, dose, frequency
   - Include both current and historical medications
   - Look for medication lists, prescriptions, orders

2. **Labs**: Test name, numeric value, unit, flag (HIGH/LOW/normal)
   - Extract all lab values mentioned
   - Note abnormal flags

3. **Conditions**: Current diagnoses
   - Include ICD-10 codes if present
   - Note status (active, resolved, suspected)

4. **Prior Year Conditions**: Diagnoses from previous years
   - Look for "prior year problem list" or similar
   - Include year if mentioned

5. **Symptoms**: Patient-reported complaints
   - Include severity and duration if mentioned
   - Extract from HRA, progress notes, chief complaints

Focus on clinical accuracy. Do not infer conditions not explicitly stated."""

MAX_DOCUMENT_LENGTH = 20000


def extraction_node(state: AgentState) -> dict:
    documents = state.get("documents", [])
    patient_id = state.get("patient_id", "unknown")

    if not documents:
        logger.warning("No documents to extract from")
        return {
            "medications": [],
            "labs": [],
            "conditions": [],
            "prior_year_conditions": [],
            "symptoms": [],
            "next_step": "supervisor",
        }

    # Combine documents
    combined_text = f"Patient: {patient_id}\n\n"
    for doc in documents:
        doc_type = doc.get("type", "document")
        doc_date = doc.get("date", "")
        content = doc.get("content", "")

        combined_text += f"=== {doc_type.upper()} ({doc_date}) ===\n{content}\n\n"

    # Truncate
    if len(combined_text) > MAX_DOCUMENT_LENGTH:
        logger.warning(f"Truncating documents from {len(combined_text)} to {MAX_DOCUMENT_LENGTH} chars")
        combined_text = combined_text[:MAX_DOCUMENT_LENGTH] + "\n... [truncated]"

    logger.info(f"Extracting clinical entities from {len(documents)} documents ({len(combined_text)} chars)")

    try:
        client = get_gemini_client()
        # Use Flash for extraction
        result = client.generate_structured(
            prompt=f"Extract clinical entities from these documents:\n\n{combined_text}",
            response_schema=EXTRACTION_SCHEMA,
            model=GEMINI_FLASH_MODEL,
            system_instruction=EXTRACTION_PROMPT,
        )

        # Normalize
        medications = normalize_medications(result.get("medications", []))
        labs = normalize_labs(result.get("labs", []))
        conditions = normalize_conditions(result.get("conditions", []))
        prior_conditions = normalize_conditions(result.get("prior_year_conditions", []))
        symptoms = result.get("symptoms", [])

        logger.info(
            f"Extracted: {len(medications)} meds, {len(labs)} labs, "
            f"{len(conditions)} conditions, {len(prior_conditions)} prior, {len(symptoms)} symptoms"
        )

        return {
            "medications": medications,
            "labs": labs,
            "conditions": conditions,
            "prior_year_conditions": prior_conditions,
            "symptoms": symptoms,
            "next_step": "supervisor",
        }

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            "medications": [],
            "labs": [],
            "conditions": [],
            "prior_year_conditions": [],
            "symptoms": [],
            "error": str(e),
            "next_step": "supervisor",
        }


def normalize_medications(meds: list[dict]) -> list[dict]:
    normalized = []
    seen = set()

    for med in meds:
        name = med.get("name", "").strip().lower()
        if not name or name in seen:
            continue

        seen.add(name)
        normalized.append({
            "name": med.get("name", "").strip(),
            "name_lower": name,
            "dose": med.get("dose", ""),
            "frequency": med.get("frequency", ""),
        })

    return normalized


def normalize_labs(labs: list[dict]) -> list[dict]:
    normalized = []
    seen = set()

    for lab in labs:
        name = lab.get("name", "").strip().lower()
        if not name:
            continue

        # Skip dupes
        key = f"{name}:{lab.get('value', '')}"
        if key in seen:
            continue
        seen.add(key)

        # Ensure numeric
        value = lab.get("value")
        if value is not None:
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = None

        normalized.append({
            "name": lab.get("name", "").strip(),
            "name_lower": name,
            "value": value,
            "unit": lab.get("unit", ""),
            "flag": lab.get("flag", "normal"),
        })

    return normalized


def normalize_conditions(conditions: list[dict]) -> list[dict]:
    normalized = []
    seen = set()

    for cond in conditions:
        name = cond.get("name", "").strip().lower()
        if not name or name in seen:
            continue

        seen.add(name)
        normalized.append({
            "name": cond.get("name", "").strip(),
            "name_lower": name,
            "icd10": cond.get("icd10", ""),
            "status": cond.get("status", "active"),
            "year": cond.get("year"),
        })

    return normalized
