# One SQLite store; extraction is model-independent, evaluation is not

Upstream caches pipeline output as JSON files keyed by the PDF's basename, which collides across candidates and ignores Role and model. We replace that with one SQLite file, keyed by content, with a deliberate asymmetry between the two stages:

- **Resume identity** — SHA-256 of the raw PDF bytes (Drive's `sha256Checksum` when listing, so already-seen Drive files are skipped without downloading). A re-exported PDF is a new Resume; accepted.
- **Extracted Resume** — keyed by resume hash only. Extraction is meant to be a faithful transcription, so switching models does not re-extract; the extracting model is recorded for reference. `--re-extract` replaces it and deletes that Resume's Evaluations, which were computed from the old extraction.
- **Evaluation** — keyed by `(resume hash, role content hash, provider:model)`. The Role is identified by the content of `role.json`, `criteria.jinja` and `system_message.jinja`, so an edited Role re-evaluates instead of silently returning stale scores. The model is the judge, so a different model is a different Evaluation; results from different models coexist for comparison. The provider is part of the judge's identity because the same model name can be served at different quantisations (e.g. `ollama:gemma3:12b` vs `openrouter:google/gemma-3-12b-it`).
- **GitHub Enrichment** — not cached separately; it depends on Role, model and time, so each Evaluation stores the snapshot it was judged on. Raw GitHub API responses stay in upstream's file cache.

## Considered Options

- Keying Extracted Resume by `(resume hash, model)` — rejected: Evaluations would then also have to reference which extraction they used, for little gain over a manual `--re-extract`.
- Keying Evaluation by Role name only — rejected: editing a Role's criteria would return stale Evaluations with no visible signal.
