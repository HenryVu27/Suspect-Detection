import logging

from agents.state import AgentState

logger = logging.getLogger(__name__)


def report_node(state: AgentState) -> dict:
    """Generate analysis report from validated findings.

    This node is only reached via the analysis path (orchestrator -> load_documents
    -> extraction -> supervisor -> detection -> aggregate -> self_reflect -> report).
    Info requests are handled by answer_query_node via a separate path.
    """
    patient_id = state.get("patient_id", "Unknown")
    validated_findings = state.get("validated_findings", [])
    error = state.get("error")

    # Handle error case
    if error and not validated_findings:
        return {
            "response": f"Analysis encountered an error: {error}\n\nPlease try again or check the patient ID.",
            "next_step": "end",
        }

    # Handle no findings
    if not validated_findings:
        return {
            "response": f"**Patient Analysis: {patient_id}**\n\nNo suspect conditions or gaps were detected. The patient's documentation appears consistent.",
            "next_step": "end",
        }

    logger.info(f"Generating report with {len(validated_findings)} findings")

    # Group findings by severity
    by_severity = {"critical": [], "high": [], "medium": [], "low": []}
    for finding in validated_findings:
        severity = finding.get("severity", "low")
        by_severity.get(severity, by_severity["low"]).append(finding)

    # Build structured report
    report_lines = [
        f"## Suspect Detection Report: {patient_id}",
        "",
        f"**{len(validated_findings)} potential issues detected**",
        "",
    ]

    severity_labels = {
        "critical": "CRITICAL - Immediate Attention Required",
        "high": "HIGH PRIORITY",
        "medium": "MEDIUM PRIORITY",
        "low": "LOW PRIORITY",
    }

    for severity in ["critical", "high", "medium", "low"]:
        findings = by_severity[severity]
        if not findings:
            continue

        report_lines.append(f"### {severity_labels[severity]}")
        report_lines.append("")

        for i, finding in enumerate(findings, 1):
            finding_type = finding.get("type", "unknown")
            signal = finding.get("signal", "No details")
            confidence = finding.get("confidence", 0.5)

            # Type-specific formatting
            if finding_type == "medication_diagnosis_gap":
                med = finding.get("medication", "")
                expected = finding.get("expected_conditions", [])
                report_lines.append(f"{i}. **Medication without diagnosis**: {med}")
                report_lines.append(f"   - Expected condition: {', '.join(expected[:2])}")
                report_lines.append(f"   - *Action*: Verify if condition should be documented")

            elif finding_type == "lab_diagnosis_gap":
                lab = finding.get("lab", "")
                value = finding.get("value", "")
                expected = finding.get("expected_condition", "")
                report_lines.append(f"{i}. **Abnormal lab without diagnosis**: {lab} = {value}")
                report_lines.append(f"   - Suggests: {expected}")
                report_lines.append(f"   - *Action*: Consider adding diagnosis if clinically appropriate")

            elif finding_type == "chronic_condition_dropoff":
                condition = finding.get("condition", "")
                report_lines.append(f"{i}. **Chronic condition drop-off**: {condition}")
                report_lines.append(f"   - Documented in prior year but missing from current")
                report_lines.append(f"   - *Action*: Verify current status and update documentation")

            elif finding_type == "symptom_cluster":
                symptoms = finding.get("matching_symptoms", [])
                suggested = finding.get("suggested_condition", "")
                report_lines.append(f"{i}. **Symptom pattern detected**: {suggested}")
                report_lines.append(f"   - Matching symptoms: {', '.join(symptoms)}")
                report_lines.append(f"   - *Action*: Consider screening/workup if not already done")

            elif finding_type == "contradiction":
                report_lines.append(f"{i}. **Documentation contradiction**")
                report_lines.append(f"   - {signal}")
                report_lines.append(f"   - *Action*: Clarify and resolve conflicting information")

            else:
                report_lines.append(f"{i}. {signal}")

            if confidence < 0.7:
                report_lines.append(f"   - *(Confidence: {confidence:.0%})*")

            report_lines.append("")

    # Add summary footer
    total_high_priority = len(by_severity["critical"]) + len(by_severity["high"])
    if total_high_priority > 0:
        report_lines.append("---")
        report_lines.append(f"**{total_high_priority} high-priority items require attention.**")

    response = "\n".join(report_lines)

    return {
        "response": response,
        "next_step": "end",
    }
