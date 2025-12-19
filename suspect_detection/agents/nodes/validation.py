import logging

from agents.state import AgentState
from agents.models import FINDING_VALIDATION_SCHEMA
from agents.gemini_client import get_gemini_client
from config import GEMINI_FLASH_MODEL

logger = logging.getLogger(__name__)

MAX_REFINEMENT_ATTEMPTS = 2

VALIDATION_PROMPT = """You are a clinical validation expert. Your job is to verify that clinical findings are supported by the source documents.

For each finding, evaluate:
1. **is_supported**: Is the finding directly supported by evidence in the documents?
2. **has_hallucination**: Does the finding claim something NOT in the documents?
3. **confidence**: How confident are you in the finding? (0.0 - 1.0)
4. **issues**: What specific problems exist with this finding?
5. **suggested_fix**: How could the finding be corrected?

Be rigorous but fair. A finding can be valid even if the exact wording differs from the source.
Focus on factual accuracy, not stylistic concerns.
"""


def self_reflect_node(state: AgentState) -> dict:
    findings = state.get("findings", [])
    documents = state.get("documents", [])
    refinement_attempts = state.get("refinement_attempts", 0)
    already_validated = state.get("validated_findings", [])

    # Re-entry after refinement
    if not findings and already_validated:
        logger.info(f"Re-entry after refinement: {len(already_validated)} validated findings")
        return {
            "next_step": "report",
        }

    if not findings:
        logger.info("No findings to validate")
        return {
            "validated_findings": [],
            "findings_to_refine": [],
            "next_step": "report",
        }

    # Skip already validated
    validated_signals = {f.get("signal", "") for f in already_validated}
    findings_to_check = [f for f in findings if f.get("signal", "") not in validated_signals]

    if not findings_to_check:
        logger.info("All findings already validated")
        return {"next_step": "report"}

    logger.info(f"Validating {len(findings_to_check)} findings (attempt {refinement_attempts + 1})")

    # Doc reference
    doc_summary = ""
    for doc in documents[:5]:
        doc_type = doc.get("type", "document")
        content = doc.get("content", "")[:2000]
        doc_summary += f"\n=== {doc_type} ===\n{content}\n"

    client = get_gemini_client()
    validated = list(already_validated)  # Start with already validated
    needs_refinement = []

    for finding in findings_to_check:
        try:
            # Validate finding
            validation = client.generate_structured(
                prompt=f"""Finding to validate:
Type: {finding.get('type')}
Signal: {finding.get('signal')}
Severity: {finding.get('severity')}

Source Documents:
{doc_summary}

Validate this finding against the source documents.""",
                response_schema=FINDING_VALIDATION_SCHEMA,
                model=GEMINI_FLASH_MODEL,
                system_instruction=VALIDATION_PROMPT,
            )

            is_supported = validation.get("is_supported", True)
            has_hallucination = validation.get("has_hallucination", False)
            confidence = validation.get("confidence", 0.5)
            issues = validation.get("issues", [])

            if is_supported and not has_hallucination and confidence >= 0.6:
                validated.append({**finding, "confidence": confidence, "validated": True})
            elif confidence >= 0.3 and refinement_attempts < MAX_REFINEMENT_ATTEMPTS:
                needs_refinement.append({
                    **finding,
                    "validation_issues": issues,
                    "suggested_fix": validation.get("suggested_fix"),
                })
            else:
                # Low confidence
                logger.info(f"Dropping finding (low confidence {confidence}): {finding.get('signal', '')[:50]}")

        except Exception as e:
            logger.warning(f"Validation failed for finding: {e}")
            validated.append({**finding, "confidence": 0.5, "validated": False})

    new_validated = len(validated) - len(already_validated)
    logger.info(f"Validation: {new_validated} newly validated, {len(needs_refinement)} need refinement (total: {len(validated)})")

    if needs_refinement and refinement_attempts < MAX_REFINEMENT_ATTEMPTS:
        return {
            "validated_findings": validated,
            "findings_to_refine": needs_refinement,
            "refinement_attempts": refinement_attempts + 1,
            "next_step": "refine",
        }

    # Done
    return {
        "validated_findings": validated + needs_refinement,  # Include remaining as-is
        "findings_to_refine": [],
        "next_step": "report",
    }


def refine_node(state: AgentState) -> dict:
    findings_to_refine = state.get("findings_to_refine", [])
    documents = state.get("documents", [])

    if not findings_to_refine:
        return {"next_step": "self_reflect"}

    logger.info(f"Refining {len(findings_to_refine)} findings")

    # Doc reference
    doc_summary = ""
    for doc in documents[:5]:
        doc_type = doc.get("type", "document")
        content = doc.get("content", "")[:2000]
        doc_summary += f"\n=== {doc_type} ===\n{content}\n"

    client = get_gemini_client()
    refined_findings = []

    for finding in findings_to_refine:
        issues = finding.get("validation_issues", [])
        suggested_fix = finding.get("suggested_fix", "")

        try:
            # Refine finding
            refined_text = client.generate(
                prompt=f"""Original finding:
Type: {finding.get('type')}
Signal: {finding.get('signal')}

Issues found: {', '.join(issues)}
Suggested fix: {suggested_fix}

Source documents:
{doc_summary}

Please provide a corrected version of this finding that addresses the issues.
If the finding cannot be corrected (no evidence), respond with "INVALID".

Respond with just the corrected signal text, nothing else.""",
                model=GEMINI_FLASH_MODEL,
                system_instruction="You are refining clinical findings to be more accurate. Be concise.",
            )

            refined_text = refined_text.strip()

            if refined_text.upper() == "INVALID" or not refined_text:
                logger.info(f"Finding marked invalid: {finding.get('signal', '')[:50]}")
                continue

            refined_finding = {
                k: v for k, v in finding.items()
                if k not in ("validation_issues", "suggested_fix")
            }
            refined_finding["signal"] = refined_text
            refined_finding["refined"] = True
            refined_findings.append(refined_finding)

        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
            # Keep original
            refined_findings.append(finding)

    logger.info(f"Refined {len(refined_findings)} findings")

    # Add to validated
    existing_validated = state.get("validated_findings", [])
    return {
        "validated_findings": existing_validated + refined_findings,
        "findings_to_refine": [],
    }
