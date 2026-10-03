# Hiring Agent

A pipeline that turns a candidate's Resume into structured data, enriches it with GitHub signals, and produces an explainable Evaluation of that Resume against a chosen Role.

## Language

**Resume**:
The source document (PDF) representing one candidate, read from a Resume Source. Identified by its exact content: two files with identical bytes are the same Resume regardless of name or location; a re-exported PDF is a different Resume.
_Avoid_: CV, 履歷檔

**Extracted Resume**:
The structured data (JSON Resume shape) extracted from a Resume. Independent of any Role. A Resume has at most one Extracted Resume; re-extracting replaces it and invalidates every Evaluation of that Resume.
_Avoid_: parsed resume, resume JSON, resume data

**Role**:
A named scoring rubric — its categories, their weights, bonus and score bounds, and the prompts that encode how to score against them.
_Avoid_: JD, rubric, position

**Evaluation**:
The result of scoring one Resume against one Role: per-category scores with evidence, bonus points, deductions, strengths and areas for improvement. Identified by the Resume, the exact content of the Role, and the model that judged it together with the provider serving it — editing the Role or switching the model or provider yields a new Evaluation alongside the old one.
_Avoid_: score result, 打分結果

**Score**:
A number inside an Evaluation — a category score or the total. Never the Evaluation as a whole.

**GitHub Enrichment**:
Profile and LLM-selected project data fetched from the GitHub profile linked in a Resume, added to the input of an Evaluation. It depends on the Role, the model and the moment it was fetched, so each Evaluation keeps its own snapshot of the GitHub Enrichment it was judged on. A Resume that links a GitHub profile is never Evaluated without its GitHub Enrichment; if fetching it fails, the Evaluation fails.
_Avoid_: github data

**Resume Source**:
Where Resumes are listed and read from — a local folder or Google Drive.
_Avoid_: loader, provider (reserved for LLM providers)

**Ranking**:
The Evaluations of every Resume in a run's selected Batches, against one Role by one model, ordered by total Score. Always complete — it includes Evaluations reused from earlier runs, not just newly computed ones. Not cut to a top N.
_Avoid_: report, shortlist, leaderboard

**Batch**:
The Resumes received on one date, kept together in a Resume Source under that date (`YYYYMMDD`). A run selects an inclusive range of Batches.
_Avoid_: day folder, intake
