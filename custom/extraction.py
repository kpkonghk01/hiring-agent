"""Turn a Resume into an Extracted Resume: OCR, then one LLM call for the whole resume.

Upstream's PDFHandler asks the LLM once per section — six calls, each resending the
whole resume, and none for languages. For batch throughput this handler asks once,
with a prompt merged from upstream's section templates plus languages
(custom/prompts/resume.jinja), and keeps upstream's JSON cleanup and normalisation.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ValidationError

import pdf
from custom.llm import LLMChoice, ObservedClient, bound_to
from custom.ocr import MarkerOcr, extract_text_layer
from models import (
    Award,
    Basics,
    Education,
    JSONResume,
    Language,
    OpenAICompatibleProvider,
    Project,
    Skill,
    Work,
)

logger = logging.getLogger(__name__)

# The denoised text layer is a hint, not the Resume; keep it from crowding the prompt.
_MAX_TEXT_LAYER_CHARS = 6000
_RESUME_PROMPT = (Path(__file__).parent / "prompts" / "resume.jinja").read_text(
    encoding="utf-8"
)
_EXTRACTION_ATTEMPTS = 2  # one retry, as upstream does per section
# Fills the "{{ section_name_param }}" of upstream's extraction system message.
_ALL_SECTIONS = (
    "full resume (basics, work, education, skills, projects, awards, languages)"
)


class ResumeSections(BaseModel):
    """Structured-output schema of the single extraction call. ``basics`` is required
    so upstream's normaliser treats the reply as a whole resume, not one section."""

    basics: Basics
    work: Optional[List[Work]] = None
    education: Optional[List[Education]] = None
    skills: Optional[List[Skill]] = None
    projects: Optional[List[Project]] = None
    awards: Optional[List[Award]] = None
    languages: Optional[List[Language]] = None


class OcrPDFHandler(pdf.PDFHandler):
    """Upstream's extraction, fed with OCR text and asked in a single call."""

    def __init__(self, ocr: MarkerOcr, file_name: str):
        self._ocr = ocr
        self._file_name = file_name
        super().__init__()

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        started = time.monotonic()
        markdown = self._ocr.to_markdown(Path(pdf_path))
        logger.info("OCR of %s took %.1fs", self._file_name, time.monotonic() - started)
        if not markdown.strip():
            logger.error("OCR found no text in %s", self._file_name)
            return None
        return compose_resume_text(
            markdown, _text_layer_hint(Path(pdf_path)), self._file_name
        )

    def _extract_all_sections_separately(
        self, text_content: str
    ) -> Optional[JSONResume]:
        """One call for every section instead of upstream's one call per section."""
        prompt = self.template_manager.render_string(
            _RESUME_PROMPT, text_content=text_content
        )
        for _ in range(_EXTRACTION_ATTEMPTS):
            data = self._call_llm_for_section(
                _ALL_SECTIONS, text_content, prompt, ResumeSections
            )
            if data:
                return _to_json_resume(data)
        logger.error("Failed to extract %s", self._file_name)
        return None


def _to_json_resume(data: dict) -> JSONResume:
    """Upstream-normalised sections -> JSONResume, dropping basics that do not
    validate as upstream does."""
    fields = {key: data.get(key) for key in JSONResume.model_fields}
    try:
        fields["basics"] = (
            Basics.model_validate(fields["basics"]) if fields["basics"] else None
        )
    except ValidationError as err:
        logger.error("Discarding invalid basics section: %s", err)
        fields["basics"] = None
    fields["languages"] = [
        entry
        for entry in fields.get("languages") or []
        if isinstance(entry, dict) and entry.get("language")
    ]
    return JSONResume.model_validate(fields)


def _text_layer_hint(pdf_path: Path) -> str:
    """Best effort: the text layer is only a hint, so a PDF pdfplumber cannot read
    must not fail the extraction."""
    try:
        return extract_text_layer(pdf_path)
    except Exception as err:
        logger.warning(
            "Could not read the PDF text layer (%s); continuing with OCR only", err
        )
        return ""


def compose_resume_text(markdown: str, text_layer: str, file_name: str) -> str:
    """OCR Markdown first; the text layer and file name only as labelled gap-fillers."""
    parts = [markdown.strip()]
    if text_layer.strip():
        parts.append(
            "--- Supplementary hint: raw PDF text layer ---\n"
            "Noisy, watermarked and possibly out of reading order. Use it ONLY to fill "
            "fields missing above (e.g. a name rendered as an image); prefer the text "
            "above when they disagree.\n" + text_layer.strip()[:_MAX_TEXT_LAYER_CHARS]
        )
    parts.append(
        "--- Supplementary hint: source file name ---\n"
        "Platform exports often embed the candidate's name or target role here; use it "
        "only for fields still missing.\n" + file_name
    )
    return "\n\n".join(parts)


def extract_resume(
    ocr: MarkerOcr,
    pdf_bytes: bytes,
    file_name: str,
    choice: LLMChoice,
    client: OpenAICompatibleProvider,
) -> Optional[JSONResume]:
    """Run OCR and upstream's section extraction with the extraction model."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "resume.pdf"
        pdf_path.write_bytes(pdf_bytes)
        observed = ObservedClient(client, choice)
        with bound_to(pdf, choice, observed):
            resume = OcrPDFHandler(ocr, file_name).extract_json_from_pdf(str(pdf_path))
    observed.check()
    if resume is None and observed.first_failure:
        # pdf.py logs and swallows LLM errors; surface the cause instead of "no data".
        raise RuntimeError(f"LLM call failed: {observed.first_failure}")
    return resume
