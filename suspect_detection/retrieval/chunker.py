import re
import hashlib
from typing import Optional
from core.models import Document, Chunk


# Section delimiter
SECTION_DELIMITER = "=" * 80

# Min section size (smaller gets merged)
SMALL_SECTION_THRESHOLD = 200

# Min split content size
MIN_SPLIT_CONTENT_SIZE = 50


def _with_header(content: str, header: str) -> str:
    """Prepend header to content if header exists."""
    return f"{header}\n\n{content}" if header else content


class Chunker:
    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
        overlap_size: int = 100,
        max_tokens: int = 480,  # Buffer for special tokens
        embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.max_tokens = max_tokens
        self.embedding_model = embedding_model
        self._tokenizer = None  # Lazy loaded

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text, add_special_tokens=True))

    def _split_oversized_chunk(
        self,
        content: str,
        header: str,
        section_name: str
    ) -> list[tuple[str, str]]:
        # Check if split needed
        token_count = self._count_tokens(content)
        if token_count <= self.max_tokens:
            return [(section_name, content)]

        # Extract section content
        if header and content.startswith(header):
            section_body = content[len(header):].strip()
        else:
            section_body = content
            header = ""

        # Header token budget
        header_tokens = self._count_tokens(header) if header else 0
        available_tokens = self.max_tokens - header_tokens - 10  # Buffer for newlines

        if available_tokens < 50:
            # Header too large
            return [(section_name, content)]

        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', section_body)
        if not sentences:
            return [(section_name, content)]

        chunks = []
        current_sentences = []
        current_tokens = 0
        part_num = 1

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_tokens = self._count_tokens(sent)

            # Long sentence
            if sent_tokens > available_tokens:
                # Save current chunk
                if current_sentences:
                    chunk_body = " ".join(current_sentences)
                    chunk_content = _with_header(chunk_body, header)
                    if len(chunk_body) >= MIN_SPLIT_CONTENT_SIZE:
                        chunks.append((f"{section_name}_part{part_num}", chunk_content))
                        part_num += 1
                    current_sentences = []
                    current_tokens = 0

                # Split by words
                words = sent.split()
                word_chunk = []
                word_tokens = 0
                for word in words:
                    wt = self._count_tokens(word)
                    if word_tokens + wt > available_tokens and word_chunk:
                        chunk_body = " ".join(word_chunk)
                        chunk_content = _with_header(chunk_body, header)
                        if len(chunk_body) >= MIN_SPLIT_CONTENT_SIZE:
                            chunks.append((f"{section_name}_part{part_num}", chunk_content))
                            part_num += 1
                        word_chunk = [word]
                        word_tokens = wt
                    else:
                        word_chunk.append(word)
                        word_tokens += wt
                if word_chunk:
                    current_sentences = [" ".join(word_chunk)]
                    current_tokens = self._count_tokens(current_sentences[0])
                continue

            # Check limit
            if current_tokens + sent_tokens > available_tokens and current_sentences:
                chunk_body = " ".join(current_sentences)
                chunk_content = _with_header(chunk_body, header)
                if len(chunk_body) >= MIN_SPLIT_CONTENT_SIZE:
                    chunks.append((f"{section_name}_part{part_num}", chunk_content))
                    part_num += 1

                # Start new chunk with overlap (last 1-2 sentences)
                overlap_sents = current_sentences[-2:] if len(current_sentences) > 1 else current_sentences[-1:]
                current_sentences = overlap_sents + [sent]
                current_tokens = self._count_tokens(" ".join(current_sentences))
            else:
                current_sentences.append(sent)
                current_tokens += sent_tokens

        # Last chunk
        if current_sentences:
            chunk_body = " ".join(current_sentences)
            chunk_content = _with_header(chunk_body, header)
            if len(chunk_body) >= MIN_SPLIT_CONTENT_SIZE or not chunks:
                # Single chunk - use original name
                final_name = section_name if part_num == 1 else f"{section_name}_part{part_num}"
                chunks.append((final_name, chunk_content))

        # Single chunk fix
        if len(chunks) == 1:
            chunks[0] = (section_name, chunks[0][1])

        return chunks if chunks else [(section_name, content)]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        if doc.doc_type == "progress_note":
            return self._chunk_by_soap_sections(doc)
        elif doc.doc_type == "hra":
            return self._chunk_by_hra_sections(doc)
        elif doc.doc_type == "lab":
            return self._chunk_by_lab_panels(doc)
        elif doc.doc_type in ("cardiology_consult", "sleep_study", "other_consult", "imaging"):
            return self._chunk_by_delimited_sections(doc)
        elif doc.doc_type == "prior_year_problems":
            return self._chunk_by_delimited_sections(doc)
        else:
            return self._semantic_chunk(doc)

    def _create_chunk(
        self,
        doc: Document,
        content: str,
        index: int,
        section_name: Optional[str] = None
    ) -> Chunk:
        # Handle null dates
        date_part = doc.date
        if date_part is None:
            file_hash = hashlib.md5(doc.source_file.encode()).hexdigest()[:8]
            date_part = f"h{file_hash}"

        chunk_id = f"{doc.patient_id}_{doc.doc_type}_{date_part}_{index}"
        metadata = {
            "patient_id": doc.patient_id,
            "doc_type": doc.doc_type,
            "date": doc.date,
            "source_file": doc.source_file,
        }
        if section_name:
            metadata["section"] = section_name
        return Chunk(id=chunk_id, content=content.strip(), metadata=metadata)

    def _extract_header(self, content: str) -> tuple[str, str]:
        lines = content.split("\n")
        header_lines = []
        body_start = 0

        for i, line in enumerate(lines):
            # Header ends at delimiter or SOAP keyword
            if SECTION_DELIMITER in line or any(
                kw in line for kw in ["Subjective:", "Chief Complaint:", "CLINICAL INDICATION"]
            ):
                body_start = i
                break
            # Collect header lines
            if i < 10:
                header_lines.append(line)
            else:
                body_start = i
                break

        header = "\n".join(header_lines).strip()
        body = "\n".join(lines[body_start:]).strip()
        return header, body

    def _chunk_by_soap_sections(self, doc: Document) -> list[Chunk]:
        content = doc.content

        # Extract header for context
        header, body = self._extract_header(content)

        # SOAP section patterns
        soap_patterns = [
            (r"Chief Complaint:.*?(?=Subjective:|$)", "chief_complaint"),
            (r"Subjective:.*?(?=Objective:|$)", "subjective"),
            (r"Objective:.*?(?=Assessment/Plan:|Assessment:|$)", "objective"),
            (r"(?:Assessment/Plan:|Assessment:).*?(?=Electronically signed|$)", "assessment_plan"),
        ]

        # Extract sections
        raw_sections = []
        for pattern, section_name in soap_patterns:
            match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(0).strip()
                if section_content:
                    raw_sections.append((section_name, section_content))

        # Merge small sections
        merged_sections = []
        pending_small = None

        for section_name, section_content in raw_sections:
            if len(section_content) < self.min_chunk_size:
                # Queue for merge
                if pending_small:
                    # Combine pending
                    pending_small = (
                        f"{pending_small[0]}_{section_name}",
                        f"{pending_small[1]}\n\n{section_content}"
                    )
                else:
                    pending_small = (section_name, section_content)
            else:
                if pending_small:
                    # Prepend pending
                    merged_name = f"{pending_small[0]}_{section_name}"
                    merged_content = f"{pending_small[1]}\n\n{section_content}"
                    merged_sections.append((merged_name, merged_content))
                    pending_small = None
                else:
                    merged_sections.append((section_name, section_content))

        # Trailing small section
        if pending_small:
            if merged_sections:
                # Append to last
                last_name, last_content = merged_sections[-1]
                merged_sections[-1] = (
                    f"{last_name}_{pending_small[0]}",
                    f"{last_content}\n\n{pending_small[1]}"
                )
            else:
                # Keep
                merged_sections.append(pending_small)

        # Create chunks
        chunks = []
        for section_name, section_content in merged_sections:
            chunk_content = _with_header(section_content, header)

            # Split if needed
            split_chunks = self._split_oversized_chunk(chunk_content, header, section_name)
            for split_name, split_content in split_chunks:
                chunks.append(self._create_chunk(
                    doc, split_content, len(chunks), split_name
                ))

        # If no SOAP sections found, fall back to single chunk
        if not chunks:
            return [self._create_chunk(doc, content, 0)]

        return chunks

    def _chunk_by_sections(self, doc: Document) -> list[Chunk]:
        chunks = []
        content = doc.content

        # Extract header
        header, _ = self._extract_header(content)

        # Split by section headers (small sections are auto-merged)
        sections = self._split_by_section_headers(content)

        for section_name, section_content in sections:
            # Normalize section name
            clean_section_name = re.sub(r"\s*\([^)]*\)\s*$", "", section_name).strip()
            normalized_name = re.sub(r"[^a-z0-9]+", "_", clean_section_name.lower()).strip("_")

            # Check for section markers
            has_markers = bool(re.match(r"^\[[A-Z][A-Z\s\-_/]+\]", section_content))
            if has_markers:
                chunk_content = _with_header(section_content, header)
            else:
                chunk_content = _with_header(f"[{clean_section_name}]\n{section_content}", header)

            # Split if needed
            split_chunks = self._split_oversized_chunk(chunk_content, header, normalized_name)
            for split_name, split_content in split_chunks:
                chunks.append(self._create_chunk(
                    doc, split_content, len(chunks), split_name
                ))

        # Fallback
        if not chunks:
            return [self._create_chunk(doc, content, 0)]

        return chunks

    # Aliases
    def _chunk_by_hra_sections(self, doc: Document) -> list[Chunk]:
        return self._chunk_by_sections(doc)

    def _split_by_section_headers(self, content: str) -> list[tuple[str, str]]:
        # Section header pattern
        pattern = (
            r"={60,}\n"          # Opening delimiter (60+ equals)
            r"([A-Z][A-Z0-9\s\-_/]*(?:\s*\([^)]*\))?)\n"  # Section name with optional parenthetical
            r"={60,}\n"          # Closing delimiter
        )

        raw_sections = []
        matches = list(re.finditer(pattern, content))

        for i, match in enumerate(matches):
            section_name = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            # Clean trailing delimiters
            section_content = re.sub(r"\n={60,}\s*$", "", section_content).strip()

            if section_content:
                raw_sections.append((section_name, section_content))

        # Merge small sections
        sections = self._merge_small_sections(raw_sections)

        return sections

    def _merge_small_sections(
        self, sections: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        if not sections:
            return sections

        merged = []
        pending_small = []  # Accumulate small sections to merge forward

        for i, (name, content) in enumerate(sections):
            is_small = len(content) < SMALL_SECTION_THRESHOLD
            is_last = i == len(sections) - 1

            if is_small and not is_last:
                # Queue for merge section
                pending_small.append((name, content))
            else:
                # This section is large enough or is the last one
                if pending_small:
                    # Prepend small sections
                    combined_content_parts = []
                    combined_names = []
                    for small_name, small_content in pending_small:
                        combined_names.append(small_name)
                        combined_content_parts.append(f"[{small_name}]\n{small_content}")
                    combined_content_parts.append(f"[{name}]\n{content}")

                    # Merged name
                    merged_name = name
                    if combined_names:
                        merged_name = f"{' + '.join(combined_names)} + {name}"

                    merged.append((merged_name, "\n\n".join(combined_content_parts)))
                    pending_small = []
                elif is_small and is_last and merged:
                    prev_name, prev_content = merged[-1]
                    combined_content = f"{prev_content}\n\n[{name}]\n{content}"
                    merged[-1] = (f"{prev_name} + {name}", combined_content)
                else:
                    merged.append((name, content))

        # Edge case: all small
        if pending_small and not merged:
            # Combine all
            combined_names = [name for name, _ in pending_small]
            combined_content = "\n\n".join(
                f"[{name}]\n{content}" for name, content in pending_small
            )
            merged.append((" + ".join(combined_names), combined_content))

        return merged

    def _chunk_by_lab_panels(self, doc: Document) -> list[Chunk]:
        return self._chunk_by_sections(doc)

    def _chunk_by_delimited_sections(self, doc: Document) -> list[Chunk]:
        return self._chunk_by_sections(doc)

    def _get_overlap_text(self, text: str, target_size: int) -> str:
        if len(text) <= target_size:
            return text

        # Find sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return text[-target_size:]

        # Build overlap from end
        overlap_parts = []
        current_len = 0

        for sent in reversed(sentences):
            if current_len + len(sent) + 1 <= target_size:
                overlap_parts.insert(0, sent)
                current_len += len(sent) + 1
            elif not overlap_parts:
                # Take last portion
                return sent[-(target_size):]
            else:
                break

        return " ".join(overlap_parts)

    def _semantic_chunk(self, doc: Document) -> list[Chunk]:
        content = doc.content.strip()

        # Small enough - single chunk
        if len(content) <= self.max_chunk_size:
            return [self._create_chunk(doc, content, 0)]

        chunks = []
        # Split on paragraphs
        paragraphs = re.split(r"\n\s*\n", content)

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Exceeds max - save and start new
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(doc, current_chunk, len(chunks)))
                    # Overlap
                    if self.overlap_size > 0:
                        overlap = self._get_overlap_text(current_chunk, self.overlap_size)
                        current_chunk = overlap + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Split by sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 > self.max_chunk_size:
                            if current_chunk:
                                chunks.append(self._create_chunk(doc, current_chunk, len(chunks)))
                            current_chunk = sent
                        else:
                            current_chunk = current_chunk + " " + sent if current_chunk else sent
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        # Last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(doc, current_chunk, len(chunks)))
        elif current_chunk and chunks:
            # Append small remainder
            last_chunk = chunks[-1]
            chunks[-1] = self._create_chunk(
                doc, last_chunk.content + "\n\n" + current_chunk, len(chunks) - 1
             )
        elif current_chunk:
            # Need at least one
            chunks.append(self._create_chunk(doc, current_chunk, 0))

        return chunks if chunks else [self._create_chunk(doc, content, 0)]
