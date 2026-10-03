"""Read boss直聘 Resumes past their poisoned text layer (docs/adr/0003).

Copied from cv-agent (ocr/marker_engine.py, ocr/text_layer.py), not imported, so the
two repos stay independent. Marker force-OCR rasterises each page and OCRs the pixels;
the pdfplumber text layer, denoised, is only a secondary hint for fields Marker drops
(e.g. a name rendered as an image).
"""

import re
from pathlib import Path

# A long, spaceless, mixed letter+digit token is almost always a watermark id, not
# Resume content (real long tokens like "SpringCloudGateway" have no digits; phones
# are all digits; emails contain '@' or '.').
_WATERMARK_TOKEN = re.compile(r"^[A-Za-z0-9_\-~]{16,}$")


class MarkerOcr:
    """Marker in force-OCR mode. Models load lazily on first use (slow, multi-GB).

    Not thread-safe (pdfium); batch runs are sequential.
    """

    def __init__(self):
        self._converter = None

    def to_markdown(self, pdf_path: Path) -> str:
        from marker.output import text_from_rendered

        if self._converter is None:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            self._converter = PdfConverter(
                artifact_dict=create_model_dict(), config={"force_ocr": True}
            )
        rendered = self._converter(str(pdf_path))
        markdown, _, _ = text_from_rendered(rendered)
        return markdown


def _is_watermark_token(line: str) -> bool:
    return (
        bool(_WATERMARK_TOKEN.match(line))
        and any(c.isdigit() for c in line)
        and any(c.isalpha() for c in line)
    )


def denoise_text_layer(text: str) -> str:
    """Drop the poisoned text layer's obvious noise: single-char fragments and long
    random watermark tokens."""
    kept = [
        line
        for raw in text.splitlines()
        if len(line := raw.strip()) > 1 and not _is_watermark_token(line)
    ]
    return "\n".join(kept)


def extract_text_layer(pdf_path: Path) -> str:
    """The PDF's own text layer via pdfplumber, denoised."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return denoise_text_layer(text)
