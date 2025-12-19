# Schema for clinical entity extraction
EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "medications": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "dose": {"type": "STRING"},
                    "frequency": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "labs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "value": {"type": "NUMBER"},
                    "unit": {"type": "STRING"},
                    "flag": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "conditions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "icd10": {"type": "STRING"},
                    "status": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "prior_year_conditions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "icd10": {"type": "STRING"},
                    "year": {"type": "INTEGER"},
                },
                "required": ["name"],
            },
        },
        "symptoms": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["medications", "labs", "conditions", "symptoms"],
}

# Schema for intent classification
INTENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {
            "type": "STRING",
            "enum": [
                "analyze_patient",
                "patient_info_request",
                "list_patients",
                "clarify_patient",
                "followup_question",
                "medical_question",
                "system_help",
                "greeting",
            ],
        },
        "patient_id": {
            "type": "STRING",
            "description": "Full patient ID if present (format: XXX-YYYY-NNN)",
        },
        "partial_patient_id": {
            "type": "STRING",
            "description": "Partial/incomplete patient ID if user mentioned one",
        },
        "medical_topic": {
            "type": "STRING",
            "description": "The medical topic being asked about (for medical_question intent)",
        },
        "needs_clarification": {
            "type": "BOOLEAN",
            "description": "True if patient ID is incomplete or ambiguous",
        },
        "reasoning": {"type": "STRING"},
    },
    "required": ["intent", "reasoning"],
}

# Schema for finding validation
FINDING_VALIDATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_supported": {"type": "BOOLEAN"},
        "has_hallucination": {"type": "BOOLEAN"},
        "confidence": {"type": "NUMBER"},
        "issues": {"type": "ARRAY", "items": {"type": "STRING"}},
        "suggested_fix": {"type": "STRING"},
    },
    "required": ["is_supported", "has_hallucination", "confidence"],
}

# Schema for supervisor routing decision
SUPERVISOR_DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "next_agent": {
            "type": "STRING",
            "enum": [
                "cross_reference",
                "dropoff",
                "symptom_cluster",
                "contradiction",
                "aggregate",
                "FINISH",
            ],
        },
        "reasoning": {"type": "STRING"},
    },
    "required": ["next_agent", "reasoning"],
}
