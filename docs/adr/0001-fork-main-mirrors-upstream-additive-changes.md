# Fork: `main` mirrors upstream, customisations live additively on `custom`

This repo is a fork of `interviewstreet/hiring-agent`. `main` is a pure mirror of `upstream/main` (fast-forward only, never committed to); all customisations live on the long-lived `custom` branch, which syncs by `git merge main` rather than rebase, because it is pushed and long-lived and rebasing would force-push and re-resolve the same conflicts every sync. To keep those merges cheap, customisations are **additive**: new capability goes into new modules, upstream files (`score.py`, `config.py`, `pdf.py`, `requirements.txt`, …) get only minimal hook points, and the flat module layout is kept even though a `src/` package layout would otherwise be preferable.

## Consequences

- Some duplication or indirection is accepted where rewriting an upstream file would be cleaner — conflict surface beats elegance here.
- After each upstream sync, upstream dependency changes in `requirements.txt` must be mirrored by hand.
