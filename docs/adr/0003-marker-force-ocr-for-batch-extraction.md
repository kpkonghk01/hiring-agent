# Batch extraction reads Resumes via Marker force-OCR, not the PDF text layer

The Resumes this fork processes are exported from boss直聘, whose PDFs carry a poisoned text layer (invisible white watermark text and random tokens interleaved with the content) and often two-column layouts, so upstream's PyMuPDF text-layer extraction yields garbage. `batch.py` therefore renders each page and OCRs it with **Marker in force-OCR mode**, then passes that Markdown to upstream's unchanged per-section LLM extraction prompts. Because Marker drops regions it classifies as images (e.g. a name rendered as a graphic), a denoised pdfplumber text layer and the source filename are appended as clearly-labelled secondary hints. The approach and its OCR modules are copied from the sibling `cv-agent` project (its ADR 0001), not imported, to keep the repos independent.

## Considered Options

- **Upstream PyMuPDF text layer** — rejected: buried in ~600 watermark fragments per page on real boss直聘 exports.
- **Tesseract** — rejected: mangles Chinese and transcribes the visible watermark.
- **Vision LLM OCR** — rejected for the main path: costlier, less reproducible, token-heavy.

## Consequences

- Heavy dependency (`marker-pdf` pulls torch and multi-GB models); OCR is slow, but the Extracted Resume is cached by resume hash so each Resume is OCR'd once.
- Batch runs are sequential: Marker/pdfium is not thread-safe.
- `score.py` (single-file, upstream) keeps PyMuPDF; this only applies to `batch.py`.
- Extraction is wired by subclassing `PDFHandler` and overriding `extract_text_from_pdf`, so `pdf.py` stays untouched (ADR 0001).
