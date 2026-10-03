"""Turn a Resume into an Extracted Resume via OCR plus upstream's section prompts."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import pdf
from custom.llm import LLMChoice, bound_to
from custom.ocr import MarkerOcr, extract_text_layer
from models import JSONResume, OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# The denoised text layer is a hint, not the Resume; keep it from crowding every
# section prompt.
_MAX_TEXT_LAYER_CHARS = 6000


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
        with bound_to(pdf, choice, client):
            return OcrPDFHandler(ocr, file_name).extract_json_from_pdf(str(pdf_path))
