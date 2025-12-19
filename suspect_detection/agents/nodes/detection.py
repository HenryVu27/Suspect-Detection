import logging

from agents.state import AgentState
from config import (
    LAB_HBA1C,
    LAB_EGFR,
    LAB_BNP,
    LAB_NT_PROBNP,
    LAB_CREATININE,
    LAB_TSH,
    LAB_LDL,
    SLEEP_APNEA_MIN_MATCHES,
    HEART_FAILURE_MIN_MATCHES,
    DEPRESSION_MIN_MATCHES,
    HYPOTHYROIDISM_MIN_MATCHES,
)

logger = logging.getLogger(__name__)

# Med -> condition mapping
MED_CONDITION_MAP = {
    "metformin": ["diabetes", "type 2 diabetes", "dm", "dm2"],
    "insulin": ["diabetes", "dm", "dm1", "dm2"],
    "glipizide": ["diabetes", "type 2 diabetes", "dm2"],
    "lisinopril": ["hypertension", "heart failure", "htn", "chf"],
    "losartan": ["hypertension", "htn"],
    "amlodipine": ["hypertension", "htn"],
    "carvedilol": ["heart failure", "hypertension", "chf", "htn"],
    "metoprolol": ["heart failure", "hypertension", "atrial fibrillation", "afib"],
    "furosemide": ["heart failure", "edema", "chf"],
    "spironolactone": ["heart failure", "chf"],
    "atorvastatin": ["hyperlipidemia", "dyslipidemia", "high cholesterol"],
    "simvastatin": ["hyperlipidemia", "dyslipidemia"],
    "rosuvastatin": ["hyperlipidemia", "dyslipidemia"],
    "eliquis": ["atrial fibrillation", "afib", "dvt", "pe"],
    "apixaban": ["atrial fibrillation", "afib", "dvt"],
    "warfarin": ["atrial fibrillation", "afib", "dvt", "pe"],
    "xarelto": ["atrial fibrillation", "afib", "dvt"],
    "rivaroxaban": ["atrial fibrillation", "dvt"],
    "levothyroxine": ["hypothyroidism", "thyroid"],
    "synthroid": ["hypothyroidism", "thyroid"],
    "albuterol": ["asthma", "copd", "reactive airway"],
    "ventolin": ["asthma", "copd"],
    "omeprazole": ["gerd", "acid reflux", "peptic ulcer"],
    "pantoprazole": ["gerd", "acid reflux"],
    "gabapentin": ["neuropathy", "seizure", "nerve pain"],
    "pregabalin": ["neuropathy", "fibromyalgia"],
    "sertraline": ["depression", "anxiety", "mdd"],
    "fluoxetine": ["depression", "anxiety"],
    "escitalopram": ["depression", "anxiety"],
}

# Lab thresholds for detection
LAB_THRESHOLDS = {
    "hba1c": {"threshold": LAB_HBA1C, "op": ">=", "condition": "diabetes"},
    "a1c": {"threshold": LAB_HBA1C, "op": ">=", "condition": "diabetes"},
    "hemoglobin a1c": {"threshold": LAB_HBA1C, "op": ">=", "condition": "diabetes"},
    "egfr": {"threshold": LAB_EGFR, "op": "<", "condition": "chronic kidney disease"},
    "gfr": {"threshold": LAB_EGFR, "op": "<", "condition": "chronic kidney disease"},
    "bnp": {"threshold": LAB_BNP, "op": ">", "condition": "heart failure"},
    "nt-probnp": {"threshold": LAB_NT_PROBNP, "op": ">", "condition": "heart failure"},
    "creatinine": {"threshold": LAB_CREATININE, "op": ">", "condition": "kidney impairment"},
    "tsh": {"threshold": LAB_TSH, "op": ">", "condition": "hypothyroidism"},
    "ldl": {"threshold": LAB_LDL, "op": ">", "condition": "hyperlipidemia"},
}

# Chronic conditions
CHRONIC_CONDITIONS = [
    "diabetes", "dm", "type 2 diabetes", "type 1 diabetes",
    "hypertension", "htn", "high blood pressure",
    "ckd", "chronic kidney disease", "kidney disease",
    "heart failure", "chf", "hfref", "hfpef",
    "copd", "chronic obstructive pulmonary",
    "asthma",
    "depression", "mdd", "major depressive",
    "anxiety", "gad", "generalized anxiety",
    "cad", "coronary artery disease", "coronary disease",
    "atrial fibrillation", "afib", "a-fib",
    "hypothyroidism", "thyroid",
    "hyperlipidemia", "dyslipidemia", "high cholesterol",
    "obesity", "bmi",
    "osteoporosis",
    "dementia", "alzheimer",
]

# Symptom clusters
SYMPTOM_CLUSTERS = {
    "sleep_apnea": {
        "symptoms": ["snoring", "apnea", "gasping", "daytime sleepiness", "fatigue",
                    "morning headache", "drowsy", "tired", "excessive sleepiness"],
        "min_matches": SLEEP_APNEA_MIN_MATCHES,
        "condition": "obstructive sleep apnea",
        "severity": "medium",
    },
    "heart_failure": {
        "symptoms": ["dyspnea", "shortness of breath", "edema", "swelling",
                    "fatigue", "orthopnea", "pnd", "paroxysmal nocturnal", "leg swelling"],
        "min_matches": HEART_FAILURE_MIN_MATCHES,
        "condition": "heart failure",
        "severity": "high",
    },
    "depression": {
        "symptoms": ["sad", "depressed", "hopeless", "anhedonia", "sleep problems",
                    "insomnia", "fatigue", "appetite", "weight change", "concentration"],
        "min_matches": DEPRESSION_MIN_MATCHES,
        "condition": "depression",
        "severity": "medium",
    },
    "hypothyroidism": {
        "symptoms": ["fatigue", "weight gain", "cold intolerance", "constipation",
                    "dry skin", "hair loss", "bradycardia"],
        "min_matches": HYPOTHYROIDISM_MIN_MATCHES,
        "condition": "hypothyroidism",
        "severity": "medium",
    },
}


def _condition_matches(condition_name: str, target_conditions: list[str]) -> bool:
    condition_lower = condition_name.lower()
    for target in target_conditions:
        if target in condition_lower or condition_lower in target:
            return True
    return False


def _has_condition(conditions: list[dict], target_conditions: list[str]) -> bool:
    for cond in conditions:
        name = cond.get("name_lower", cond.get("name", "").lower())
        if _condition_matches(name, target_conditions):
            return True
    return False


def cross_reference_node(state: AgentState) -> dict:
    medications = state.get("medications", [])
    labs = state.get("labs", [])
    conditions = state.get("conditions", [])

    findings = []

    # Medication gaps
    for med in medications:
        med_name = med.get("name_lower", med.get("name", "").lower())
        expected = MED_CONDITION_MAP.get(med_name, [])

        if expected and not _has_condition(conditions, expected):
            findings.append({
                "type": "medication_diagnosis_gap",
                "severity": "high",
                "medication": med.get("name"),
                "expected_conditions": expected,
                "signal": f"Patient takes {med.get('name')} but no {'/'.join(expected[:2])} documented",
                "strategy": "cross_reference",
            })

    # Lab gaps
    for lab in labs:
        lab_name = lab.get("name_lower", lab.get("name", "").lower())
        value = lab.get("value")

        for threshold_name, rule in LAB_THRESHOLDS.items():
            if threshold_name not in lab_name:
                continue

            if value is None:
                continue

            threshold_met = False
            if rule["op"] == ">=" and value >= rule["threshold"]:
                threshold_met = True
            elif rule["op"] == "<" and value < rule["threshold"]:
                threshold_met = True
            elif rule["op"] == ">" and value > rule["threshold"]:
                threshold_met = True

            if threshold_met:
                expected_condition = rule["condition"]
                if not _has_condition(conditions, [expected_condition]):
                    findings.append({
                        "type": "lab_diagnosis_gap",
                        "severity": "high",
                        "lab": lab.get("name"),
                        "value": value,
                        "unit": lab.get("unit", ""),
                        "threshold": f"{rule['op']} {rule['threshold']}",
                        "expected_condition": expected_condition,
                        "signal": f"{lab.get('name')} = {value} suggests {expected_condition} but not documented",
                        "strategy": "cross_reference",
                    })

    logger.info(f"Cross-reference found {len(findings)} gaps")

    return {
        "findings": findings,
        "completed_strategies": ["cross_reference"],
    }


def dropoff_node(state: AgentState) -> dict:
    prior_conditions = state.get("prior_year_conditions", [])
    current_conditions = state.get("conditions", [])

    findings = []

    current_names = [
        cond.get("name_lower", cond.get("name", "").lower())
        for cond in current_conditions
    ]

    for prior in prior_conditions:
        prior_name = prior.get("name_lower", prior.get("name", "").lower())

        # Only check chronic conditions
        is_chronic = any(keyword in prior_name for keyword in CHRONIC_CONDITIONS)
        if not is_chronic:
            continue

        # Check if missing
        found = any(
            prior_name in curr or curr in prior_name
            for curr in current_names
        )

        if not found:
            findings.append({
                "type": "chronic_condition_dropoff",
                "severity": "medium",
                "condition": prior.get("name"),
                "icd10": prior.get("icd10", ""),
                "year": prior.get("year"),
                "signal": f"{prior.get('name')} documented in prior year but missing from current - verify status",
                "strategy": "dropoff",
            })

    logger.info(f"Dropoff detection found {len(findings)} gaps")

    return {
        "findings": findings,
        "completed_strategies": ["dropoff"],
    }


def symptom_cluster_node(state: AgentState) -> dict:
    symptoms = state.get("symptoms", [])
    conditions = state.get("conditions", [])

    findings = []
    symptom_lower = [s.lower() for s in symptoms]

    for cluster_name, cluster in SYMPTOM_CLUSTERS.items():
        # Skip documented conditions
        if _has_condition(conditions, [cluster["condition"]]):
            continue

        # Count matches
        matches = []
        for symptom in cluster["symptoms"]:
            for patient_symptom in symptom_lower:
                if symptom in patient_symptom:
                    matches.append(symptom)
                    break

        if len(matches) >= cluster["min_matches"]:
            findings.append({
                "type": "symptom_cluster",
                "severity": cluster["severity"],
                "cluster": cluster_name,
                "matching_symptoms": matches,
                "suggested_condition": cluster["condition"],
                "signal": f"Symptoms ({', '.join(matches)}) suggest possible {cluster['condition']}",
                "strategy": "symptom_cluster",
            })

    logger.info(f"Symptom cluster found {len(findings)} potential conditions")

    return {
        "findings": findings,
        "completed_strategies": ["symptom_cluster"],
    }


def contradiction_node(state: AgentState) -> dict:
    medications = state.get("medications", [])
    conditions = state.get("conditions", [])

    findings = []

    # Resolved conditions with active meds
    for cond in conditions:
        status = cond.get("status", "active").lower()
        if status not in ["resolved", "inactive", "remission"]:
            continue

        cond_name = cond.get("name_lower", cond.get("name", "").lower())

        # Find meds treating this condition
        for med in medications:
            med_name = med.get("name_lower", med.get("name", "").lower())
            expected_conditions = MED_CONDITION_MAP.get(med_name, [])

            if any(exp in cond_name or cond_name in exp for exp in expected_conditions):
                findings.append({
                    "type": "contradiction",
                    "severity": "medium",
                    "condition": cond.get("name"),
                    "condition_status": status,
                    "medication": med.get("name"),
                    "signal": f"{cond.get('name')} marked as {status} but {med.get('name')} still prescribed",
                    "strategy": "contradiction",
                })

    logger.info(f"Contradiction detection found {len(findings)} issues")

    return {
        "findings": findings,
        "completed_strategies": ["contradiction"],
    }


def aggregate_findings_node(state: AgentState) -> dict:
    logger.info(f"Detection complete: {len(state.get('findings', []))} findings from {len(state.get('completed_strategies', []))} strategies")
    return {}
