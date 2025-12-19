def build_patient_context(state: dict) -> str:
    patient_id = state.get("patient_id", "Unknown")
    meds = state.get("medications", [])
    labs = state.get("labs", [])
    conditions = state.get("conditions", [])
    prior_conditions = state.get("prior_year_conditions", [])
    symptoms = state.get("symptoms", [])
    findings = state.get("validated_findings", []) or state.get("findings", [])

    parts = [f"**Patient: {patient_id}**\n"]

    if meds:
        med_list = "\n".join(
            f"- {m.get('name', '')} {m.get('dose', '')} {m.get('frequency', '')}".strip()
            for m in meds
        )
        parts.append(f"**Medications ({len(meds)}):**\n{med_list}\n")

    if conditions:
        cond_list = "\n".join(
            f"- {c.get('name', '')} ({c.get('status', 'active')})"
            for c in conditions
        )
        parts.append(f"**Current Conditions ({len(conditions)}):**\n{cond_list}\n")

    if prior_conditions:
        prior_list = "\n".join(f"- {c.get('name', '')}" for c in prior_conditions)
        parts.append(f"**Prior Year Conditions ({len(prior_conditions)}):**\n{prior_list}\n")

    if labs:
        # Show abnormal labs first, then some normal ones
        abnormal_labs = [l for l in labs if l.get('flag', '').lower() in ('high', 'low', 'abnormal')]
        normal_labs = [l for l in labs if l not in abnormal_labs]

        lab_lines = []
        for l in abnormal_labs[:10]:
            lab_lines.append(f"- **{l.get('name', '')}**: {l.get('value', '')} {l.get('unit', '')} [{l.get('flag', '')}]")
        for l in normal_labs[:5]:
            lab_lines.append(f"- {l.get('name', '')}: {l.get('value', '')} {l.get('unit', '')}")

        if lab_lines:
            parts.append(f"**Labs ({len(labs)} total, showing key values):**\n" + "\n".join(lab_lines) + "\n")

    if symptoms:
        symp_list = "\n".join(f"- {s}" for s in symptoms[:10])
        parts.append(f"**Symptoms ({len(symptoms)}):**\n{symp_list}\n")

    if findings:
        findings_list = "\n".join(
            f"- [{f.get('severity', 'medium')}] {f.get('signal', '')}"
            for f in findings[:5]
        )
        parts.append(f"**Detection Findings ({len(findings)}):**\n{findings_list}\n")

    return "\n".join(parts)
