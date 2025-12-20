import os
import re
from glob import glob
from typing import Optional
from core.models import Document


class DocumentLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_patient_documents(self, patient_id: str) -> list[Document]:
        patient_dir = os.path.join(self.base_path, patient_id)
        documents = []

        for file_path in glob(os.path.join(patient_dir, "*.txt")):
            content = self._read_file(file_path)
            doc = Document(
                content=content,
                patient_id=patient_id,
                doc_type=self._infer_doc_type(file_path),
                date=self._extract_date(file_path),
                source_file=file_path
            )
            documents.append(doc)

        return documents

    def _read_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _infer_doc_type(self, file_path: str) -> str:
        filename = os.path.basename(file_path).lower()

        if "progress_note" in filename:
            return "progress_note"
        elif "lab_results" in filename or "lab" in filename:
            return "lab"
        elif "hra" in filename:
            return "hra"
        elif "cardiology" in filename:
            return "cardiology_consult"
        elif "sleep_study" in filename or "polysomnography" in filename:
            return "sleep_study"
        elif "ct_" in filename or "mri_" in filename or "xray" in filename:
            return "imaging"
        elif "prior_year" in filename or "problem_list" in filename:
            return "prior_year_problems"
        elif "consult" in filename:
            return "other_consult"
        else:
            return "other"

    def _extract_date(self, file_path: str) -> Optional[str]:
        filename = os.path.basename(file_path)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        if match:
            return match.group(1)
        return None

    def list_patients(self) -> list[str]:
        patients = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path) and item.startswith("CVD-"):
                patients.append(item)
        return sorted(patients)
