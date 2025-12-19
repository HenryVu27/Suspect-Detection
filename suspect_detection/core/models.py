from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    content: str
    patient_id: str
    doc_type: str
    date: Optional[str]
    source_file: str


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict
