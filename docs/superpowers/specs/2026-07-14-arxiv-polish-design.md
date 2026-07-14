# Conservative arXiv Polish Design

## Goal

Prepare the completed assistive-VQA audit for arXiv without changing its
experiments, numerical results, statistical protocol, or claim boundary.

## Manuscript changes

- Retitle the paper to **Selective Reliability on VizWiz: A Risk-Controlled
  Audit of Two Public Vision--Language Pipelines**.
- Rewrite the abstract into a clearer problem, protocol, result, and limitation
  sequence while retaining the exact confirmatory guarantee and reported
  outcomes.
- Tighten repetitive prose in the introduction, results, limitations, and
  conclusion.
- Keep confirmatory certification results visibly separate from exploratory
  calibration, corruption, grounding, acquisition, and qualitative analyses.
- Standardize references to the evaluated systems as the base pipeline and the
  public tuned pipeline. Do not imply that the checkpoint difference identifies
  a causal fine-tuning effect. This applies only to evaluated-system labels;
  preserve the general fine-tuning discussion and every provenance and
  causal-scope caveat.
- Preserve every equation, generated result macro, model and dataset revision,
  privacy qualification, and the existing minimal AI-assistance disclosure.
- Make no code-availability claim because the repository has no public remote.
- Keep `paper/references.bib` byte-identical unless the build reports a concrete
  bibliographic defect.

## arXiv source package

Create a self-contained upload snapshot containing only:

- `main.tex`
- `references.bib`
- `paper_results.tex`
- `risk_coverage_natural.pdf`

The snapshot's `main.tex` will use local filenames rather than parent-directory
paths. Raw predictions, CSV files, review images, provenance JSON, the built PDF,
and unused figures will not be included. Package the four files as
`paper/arxiv-source.zip` after a clean build from the staged directory. The staged
manuscript must equal canonical `paper/main.tex` except for the two exact path
substitutions needed to localize the generated macros and included figure.

## Repository changes

- Edit `paper/main.tex` and, only if normalization is needed, `paper/references.bib`.
- Rebuild `paper/main.pdf` deterministically.
- Refresh only the manuscript PDF byte count and SHA-256 fields in
  `artifacts/run_provenance.json`.
- Add concise arXiv packaging instructions to `README.md`.
- Add the final upload archive at `paper/arxiv-source.zip`.

No inference, analysis, metrics, figures, tests, or generated result macros will
be changed.

## Verification

1. Record the SHA-256 of `artifacts/analysis/paper_results.tex` before editing and
   require the same hash after all manuscript and packaging work.
2. Guard the equation environments, pinned revisions, hard-coded protocol values,
   certification conditions, privacy qualification, and AI-assistance disclosure
   against unintended source changes.
3. From clean auxiliary state, build the canonical manuscript twice with Tectonic
   using the recorded `SOURCE_DATE_EPOCH`; require identical PDF SHA-256 values.
4. Scan the logs for LaTeX errors, undefined citations, and undefined references.
5. Extract the upload archive to a temporary directory and build it independently.
6. If an arXiv-compatible local TeX Live toolchain is available, also build the
   extracted source with PDFLaTeX and BibTeX. Otherwise, report local Tectonic
   verification honestly and make arXiv's generated AutoTeX preview a required
   manual gate before finalizing the submission.
7. Confirm the upload source contains no `../` paths, differs from canonical
   `paper/main.tex` only by the two expected path substitutions, and contains exactly
   the four required
   files.
8. Run `uv run pytest` and `uv run ruff check .` as repository-wide regression
   gates, not as evidence for manuscript claim preservation.
9. Confirm the PDF has no placeholder text or unresolved citations.
10. Verify the recorded PDF size and SHA-256 against the rebuilt artifact.

## Acceptance criteria

- The paper reads as a standalone preprint rather than an application work sample.
- The abstract accurately reports that mean VQA improved, the entirely-wrong rate
  fell, substantive answering also fell, and neither natural-regime policy passed
  certification; it does not imply demonstrated user utility.
- The source archive builds without relying on files outside the archive.
- Two clean canonical builds at the recorded epoch produce identical PDF hashes.
- The generated result-macro file is byte-identical before and after the polish.
- The existing statistical, reproducibility, privacy, and authorship boundaries
  remain intact.
- The local handoff states whether arXiv's AutoTeX preview has actually been checked;
  if not, preview inspection remains a required submission step.
