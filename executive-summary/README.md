# Executive Summary

LaTeX source for the separate executive summary deliverable.

The reference submission repository keeps the assessed PDFs under `report/` as
`report.pdf` and `summary.pdf`. This source tree mirrors that deliverable layout:
building here writes the finished summary to `../report/summary.pdf`, while keeping the
summary source beside the main `report/` folder.

The writing brief lives in
[`../research_notes/writing/exec-summary.md`](../research_notes/writing/exec-summary.md).
The summary is a standalone document for a broader audience, has a hard limit of 1,000
words, and should be written after the dissertation text is stable enough to derive from.
The visible layout follows the reference summary: compact all-caps title block, no
separate "Executive Summary" heading, and small two-column section heads.

## Build

From the repository root:

```sh
make summary
```

From this directory:

```sh
make
```

Both commands produce:

```text
report/summary.pdf
```

Use `make rebuild` to force a full rebuild and `make watch` while editing. The
`make texcount` target is included for the final submission pass; it requires `texcount`
to be installed or available on `PATH`.

If the final text needs citations, uncomment the bibliography lines at the end of
`summary.tex` and use only keys already present in `../report/construction/refs.bib`.

The reference summary does not show a front-cover word count. The local checklist currently
says to put the count on both documents, while `REQUIREMENTS.md` is less explicit about the
separate summary. `summary.tex` therefore hides the word-count line by default; set
`\showwordcounttrue` near the top of the file if the handbook or Sophie confirms it is
required.
