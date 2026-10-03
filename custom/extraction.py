"""Turn a Resume into an Extracted Resume via OCR plus upstream's section prompts."""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

import pdf
from custom.llm import LLMChoice, ObservedClient, bound_to
from custom.ocr import MarkerOcr, extract_text_layer
from models import JSONResume, Language, OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# The denoised text layer is a hint, not the Resume; keep it from crowding every
# section prompt.
_MAX_TEXT_LAYER_CHARS = 6000
# Upstream extracts six sections but never languages, which the Role's Cantonese and
# English bonuses depend on.
_LANGUAGES_PROMPT = (Path(__file__).parent / "prompts" / "languages.jinja").read_text(
    encoding="utf-8"
)
_SECTION_ATTEMPTS = 2  # upstream retries a failed section once


class LanguagesSection(BaseModel):
    languages: Optional[List[Language]] = None


class OcrPDFHandler(pdf.PDFHandler):
    """Upstream's per-section extraction, fed with OCR text instead of the text layer."""

    def __init__(self, ocr: MarkerOcr, file_name: str):
        self._ocr = ocr
        self._file_name = file_name
        super().__init__()

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        markdown = self._ocr.to_markdown(Path(pdf_path))
        if not markdown.strip():
            logger.error("OCR found no text in %s", self._file_name)
            return None
        return compose_resume_text(
            markdown, _text_layer_hint(Path(pdf_path)), self._file_name
        )

    def _extract_all_sections_separately(
        self, text_content: str
    ) -> Optional[JSONResume]:
        """Upstream's sections plus languages. Like upstream, a section that keeps
        failing aborts the extraction, so it is retried next run rather than cached
        without languages."""
        resume = super()._extract_all_sections_separately(text_content)
        if resume is None:
            return None
        languages = self._extract_languages(text_content)
        if languages is None:
            logger.error("Failed to extract languages section; aborting extraction")
            return None
        return JSONResume.model_validate(
            {**resume.model_dump(), "languages": languages}
        )

    def _extract_languages(self, text_content: str) -> Optional[List[dict]]:
        prompt = self.template_manager.render_string(
            _LANGUAGES_PROMPT, text_content=text_content
        )
        for _ in range(_SECTION_ATTEMPTS):
            data = self._call_llm_for_section(
                "languages", text_content, prompt, LanguagesSection
            )
            if data is not None:
                stated = data.get("languages") or []
                return [l for l in stated if isinstance(l, dict) and l.get("language")]
        return None


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
