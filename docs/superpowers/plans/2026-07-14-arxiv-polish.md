# arXiv Polish Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a clearer standalone preprint and a verified, self-contained arXiv upload archive without changing the audit's experiments, numbers, or claim boundary.

**Architecture:** Keep `paper/main.tex` as the canonical manuscript. Make only targeted prose edits, rebuild the tracked PDF twice with the recorded epoch, update its coupled provenance fields, and create a four-file upload snapshot whose paths are rewritten only in temporary staging.

**Tech Stack:** LaTeX, Tectonic 0.16.9, PowerShell, Git, Python/uv, pytest, Ruff

---

## File map

- Modify: `paper/main.tex` — canonical manuscript prose and headings.
- Modify: `paper/main.pdf` — deterministic build of the polished manuscript.
- Modify: `artifacts/run_provenance.json` — rebuilt PDF size and SHA-256 only.
- Modify: `README.md` — concise arXiv archive and preview instructions.
- Create: `paper/arxiv-source.zip` — self-contained four-file upload snapshot.
- Preserve byte-for-byte: `artifacts/analysis/paper_results.tex` and every analysis artifact.
- Leave unchanged unless a concrete bibliographic defect is found: `paper/references.bib`.

## Chunk 1: Manuscript and canonical artifact

### Task 1: Lock invariants and recover the pinned renderer

**Files:**
- Verify: `artifacts/analysis/paper_results.tex`
- Verify: `paper/main.pdf`
- Use temporarily: `%TEMP%\sash-tectonic-0.16.9\tectonic.exe`

- [ ] **Step 1: Confirm the starting worktree contains only the committed plan**

Run:

```powershell
git status --short
git log -3 --oneline
```

Expected: no uncommitted files; the design and plan commits are at the tip.

- [ ] **Step 2: Lock the generated result-macro hash**

Run:

```powershell
$macroHash = (Get-FileHash artifacts/analysis/paper_results.tex -Algorithm SHA256).Hash.ToLowerInvariant()
if ($macroHash -ne 'f1b95c5c84e4f01d12a1adf5ec14dd1fa44b70a55c73854e27e6da521dad9bb4') {
  throw "Unexpected paper_results.tex hash: $macroHash"
}
```

Expected: no output and exit code 0.

- [ ] **Step 3: Download the exact renderer recorded in provenance**

Run:

```powershell
$toolDir = Join-Path $env:TEMP 'sash-tectonic-0.16.9'
$zip = Join-Path $env:TEMP 'tectonic-0.16.9-x86_64-pc-windows-msvc.zip'
New-Item -ItemType Directory -Force $toolDir | Out-Null
Invoke-WebRequest 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-pc-windows-msvc.zip' -OutFile $zip
$zipHash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($zipHash -ne '131a24604785a9600989a3d91225f597df52ac06f00aeffe86fd529f99ee5cdd') {
  throw "Unexpected Tectonic archive hash: $zipHash"
}
Expand-Archive $zip $toolDir -Force
& (Join-Path $toolDir 'tectonic.exe') --version
```

Expected: `Tectonic 0.16.9`. The binary stays outside the repository.

### Task 2: Apply the conservative manuscript polish

**Files:**
- Modify: `paper/main.tex:116-184`
- Modify: `paper/main.tex:396-482`
- Modify: `paper/main.tex:553-596`

- [ ] **Step 1: Verify the expected source anchors before editing**

Run:

```powershell
rg -n "Does a Public VizWiz|This study contributes|Primary outcome|Calibration and shift|Grounding and acquisition|We asked whether" paper/main.tex
```

Expected: each current title, heading, and paragraph anchor appears once.

- [ ] **Step 2: Replace the title and abstract**

Use `apply_patch` to replace the title with:

```tex
\title{Selective Reliability on VizWiz:\\
\large A Risk-Controlled Audit of Two Public Vision--Language Pipelines}
```

Replace the complete abstract with:

```tex
\begin{abstract}
Model adaptation can improve average visual-question-answering accuracy without
making a system reliable about when to answer.  We compare a base
vision--language model with a public checkpoint presented as VizWiz-tuned, asking
whether either can issue substantive answers while keeping the entirely-wrong
rate at or below \TargetRisk{}.  We use a fixed fit/certification/test split
(\FitFraction/\CertFraction/\TestFraction).  For each pipeline, the fit set
selects a threshold on mean generated-answer token log-likelihood; the independent
certification set then applies a one-sided Clopper--Pearson bound with
Bonferroni-adjusted $\delta=\PrimaryFailureProb$ per policy.  Under i.i.d.
question sampling and assuming both frozen checkpoints are independent of the
certification labels, Bonferroni bounds the probability of any false certification
in this two-policy family by \FamilyFailureProb{}.  Thresholds and decisions are
frozen before test evaluation.  On
\TestN{} untouched natural questions, the public tuned checkpoint raises mean
VQA from \BaseVQA{} to \TunedVQA{} and lowers the entirely-wrong rate from
\BaseWrongRate{} to \TunedWrongRate{}, while its substantive-answer rate falls
from \BaseSubstantiveRate{} to \TunedSubstantiveRate{}.  However, the
certification upper bounds (\BaseCertUpper{} base; \TunedCertUpper{} tuned) both
exceed the \TargetRisk{} target.  Neither natural-regime policy is therefore
deployed, and prescribed test coverage is \BaseTestCoverage{} for the base
pipeline and \TunedTestCoverage{} for the tuned pipeline.  Exploratory
calibration, corruption, grounding, and acquisition analyses do not change this
decision.  These results compare two public pipelines on one assistive-VQA
distribution; they neither identify fine-tuning as the cause nor establish
broader safety or user benefit.
\end{abstract}
```

- [ ] **Step 3: Tighten the introduction's contribution and question statements**

Replace the contribution list and final introduction paragraph with:

```tex
Selective VQA makes this trade-off explicit through risk and coverage
\citep{whitehead2022reliablevqa,Dancette_2023_CVPR}.  This study contributes:
\begin{itemize}
  \item a paired audit of a base pipeline and a public checkpoint presented as
        VizWiz-tuned, using identical questions, prompts, decoding, scoring, and
        split assignments;
  \item an independent fit/certify/test protocol that prevents test-time threshold
        tuning and reports an exact one-sided certification decision; and
  \item exploratory grounding and evidence-acquisition diagnostics that remain
        separate from the confirmatory certificate.
\end{itemize}

The central question is whether either pipeline retains useful answer coverage
under a fixed, auditable risk gate---not merely whether their average scores differ.
```

- [ ] **Step 4: Clarify confirmatory and exploratory result headings**

Apply these literal replacements:

```text
\subsection{Primary outcome}
=> \subsection{Confirmatory outcome}

\subsection{Calibration and shift}
=> \subsection{Exploratory calibration and shift}

\subsection{Grounding and acquisition}
=> \subsection{Exploratory grounding and acquisition}
```

- [ ] **Step 5: Simplify the confirmatory-result interpretation without dropping counts**

Replace the prose from `Table~\ref{tab:primary} records` through the sentence ending
`post-hoc certificate.` with:

```tex
Table~\ref{tab:primary} records the only confirmatory certification checks.
Neither natural-regime policy meets the numerical gate: the base and tuned CP upper
bounds are \BaseCertUpper{} and \TunedCertUpper{}, both above \TargetRisk{}.
The base certification and deployed-test counts are
\BaseCertErrorsAccepted{} and \BaseTestErrorsAccepted{} errors/accepted; the tuned
counts are \TunedCertErrorsAccepted{} and \TunedTestErrorsAccepted{}.
The failed frozen thresholds would have accepted \BaseCandidateErrorsAccepted{}
and \TunedCandidateErrorsAccepted{} errors/answers on test, respectively
(\BaseCandidateCoverage{} coverage at \BaseCandidateRisk{} empirical risk and
\TunedCandidateCoverage{} at \TunedCandidateRisk{}).  These candidate outcomes are
diagnostics, not certificates; in particular, zero errors among 35 tuned answers
cannot repair a bound learned from only 11 accepted certification examples.
Because both policies fail, deployed test coverage is zero for both.  The paired
coverage difference is \CoverageDelta{} with a 95\% percentile-bootstrap interval
[\CoverageDeltaLow{}, \CoverageDeltaHigh{}]; this degenerate interval follows
mechanically and does not establish equivalence.  The public tuned checkpoint
therefore improves average task utility but does not demonstrate certified
selective reliability under the prespecified protocol.
```

- [ ] **Step 6: Tighten one limitations sentence and the conclusion**

Replace the opening of `\paragraph{Statistical scope.}` with:

```tex
The exact binomial bound relies on an independently selected threshold and i.i.d.
question sampling from the target distribution for the frozen policy; we do not
claim that exchangeability alone is sufficient.  A simple question-level split does not
protect against unidentified near duplicates or future distribution shift.
```

Replace the complete conclusion prose with:

```tex
We audited whether a public checkpoint presented as VizWiz-tuned improves
selective reliability over its base pipeline at a prespecified entirely-wrong risk
target.  The tuned checkpoint improves average VQA and reduces entirely-wrong
outputs, but neither natural-regime likelihood policy passes independent
certification.  Both therefore have zero certified answer coverage under the
protocol.  Sparse tuning provenance prevents attributing the observed differences
causally to fine-tuning.  The findings apply to the frozen VizWiz evaluation and
the stated fit/certify/test procedure; broader safety and user-impact claims
require separate evidence.
```

- [ ] **Step 7: Run static manuscript guards**

Run:

```powershell
$text = Get-Content paper/main.tex -Raw
if ($text -match 'Does a Public VizWiz Checkpoint') { throw 'Old title remains' }
if ($text -notmatch 'Selective Reliability on VizWiz') { throw 'New title missing' }
if ($text -notmatch '\\subsection\{Confirmatory outcome\}') { throw 'Confirmatory heading missing' }
$baseline = ((git show HEAD:paper/main.tex) -join "`n") -replace "`r`n", "`n"
$normalizedText = $text -replace "`r`n", "`n"
$equationPattern = '(?s)\\begin\{equation\}.*?\\end\{equation\}'
$beforeEquations = @([regex]::Matches($baseline, $equationPattern) | ForEach-Object Value)
$afterEquations = @([regex]::Matches($normalizedText, $equationPattern) | ForEach-Object Value)
if ($beforeEquations.Count -ne $afterEquations.Count) { throw 'Equation count changed' }
for ($i = 0; $i -lt $beforeEquations.Count; $i++) {
  if ($beforeEquations[$i] -cne $afterEquations[$i]) { throw "Equation $i changed" }
}
$protected = @(
  '\newcommand{\TargetRisk}{0.10}',
  '\newcommand{\FamilyFailureProb}{0.05}',
  '\newcommand{\PrimaryFailureProb}{0.025}',
  '\newcommand{\SplitSeed}{20260712}',
  'at commit \texttt{0c351dd}',
  'at commit \texttt{984042c}',
  '\texttt{lmms-lab/VizWiz-VQA}',
  '\texttt{d428a2d}',
  'Transformers 4.57.6 and PyTorch 2.6',
  '$\delta=\PrimaryFailureProb$ per policy.'
)
foreach ($anchor in $protected) {
  if (-not $text.Contains($anchor)) { throw "Protected source anchor changed: $anchor" }
}
if ($text -notmatch 'The four displayed qualitative examples were\s+screened for private or identifying content before inclusion\.') { throw 'Privacy qualification changed' }
if ($text -notmatch 'Generative AI assisted with implementation, analysis, and drafting\.\s+The author\s+verified all sources, artifacts, and claims and remains responsible for the work\.') { throw 'Disclosure changed' }
git diff --check
git diff -- paper/main.tex
```

Expected: no guard fails; the diff contains only the approved prose and heading changes.

- [ ] **Step 8: Obtain a claim-preservation review**

Dispatch a fresh reviewer with only `paper/main.tex`, its Git diff, the design spec,
and this instruction: verify that every number, equation, guarantee, limitation,
and confirmatory/exploratory boundary is preserved. Fix only concrete issues and
repeat until approved.

### Task 3: Rebuild twice and update coupled provenance

**Files:**
- Modify: `paper/main.pdf`
- Modify: `artifacts/run_provenance.json:140,166-173`
- Verify: `paper/main.log`

- [ ] **Step 1: Run two clean canonical builds**

Run from the repository root:

```powershell
$tectonic = Join-Path $env:TEMP 'sash-tectonic-0.16.9\tectonic.exe'
$env:SOURCE_DATE_EPOCH = '1783900800'
Push-Location paper
Remove-Item main.aux,main.bbl,main.blg,main.log,main.out,main.pdf -Force -ErrorAction SilentlyContinue
& $tectonic main.tex --keep-logs --color never
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'First Tectonic build failed' }
$hash1 = (Get-FileHash main.pdf -Algorithm SHA256).Hash
Remove-Item main.aux,main.bbl,main.blg,main.log,main.out,main.pdf -Force -ErrorAction SilentlyContinue
& $tectonic main.tex --keep-logs --color never
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'Second Tectonic build failed' }
$hash2 = (Get-FileHash main.pdf -Algorithm SHA256).Hash
if ($hash1 -ne $hash2) { Pop-Location; throw "Nondeterministic PDF: $hash1 != $hash2" }
Pop-Location
```

Expected: both builds succeed and the hashes match.

- [ ] **Step 2: Scan the retained build log and enforce eight pages**

Run:

```powershell
$bad = Select-String -Path paper/main.log -Pattern 'undefined references|Citation .* undefined|LaTeX Error|Package .* Error' -CaseSensitive:$false
if ($bad) { $bad; throw 'LaTeX log contains blocking diagnostics' }
$log = Get-Content paper/main.log -Raw
if ($log -notmatch 'Output written on .*\(8 pages') { throw 'Expected an 8-page PDF' }
```

Then extract and inspect the built PDF text without modifying project dependencies:

```powershell
@'
from pypdf import PdfReader

reader = PdfReader("paper/main.pdf")
assert len(reader.pages) == 8, f"expected 8 pages, found {len(reader.pages)}"
text = "\n".join(page.extract_text() or "" for page in reader.pages)
upper = text.upper()
for token in ("TODO", "TBD", "PLACEHOLDER", "FIXME"):
    assert token not in upper, f"placeholder remains in PDF: {token}"
assert "Selective Reliability on VizWiz" in text
'@ | uv run --with pypdf python -
```

Expected: no blocking diagnostics, eight pages, no placeholder token, and the new title in extracted PDF text.

- [ ] **Step 3: Record the rebuilt artifact values**

Run:

```powershell
$pdf = Get-Item paper/main.pdf
$pdfHash = (Get-FileHash $pdf.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
"pdf_bytes=$($pdf.Length)"
"paper_pdf=$pdfHash"
```

Use `apply_patch` to update only `integrity.sha256.paper_pdf` and
`manuscript.pdf_bytes` in `artifacts/run_provenance.json`.

- [ ] **Step 4: Verify provenance and repository gates**

Run:

```powershell
$p = Get-Content artifacts/run_provenance.json -Raw | ConvertFrom-Json
$actualHash = (Get-FileHash paper/main.pdf -Algorithm SHA256).Hash.ToLowerInvariant()
$actualBytes = (Get-Item paper/main.pdf).Length
if ($p.integrity.sha256.paper_pdf -ne $actualHash) { throw 'PDF hash provenance mismatch' }
if ($p.manuscript.pdf_bytes -ne $actualBytes) { throw 'PDF byte provenance mismatch' }
if ($p.manuscript.pages -ne 8) { throw 'Unexpected recorded page count' }
$macroHash = (Get-FileHash artifacts/analysis/paper_results.tex -Algorithm SHA256).Hash.ToLowerInvariant()
if ($macroHash -ne 'f1b95c5c84e4f01d12a1adf5ec14dd1fa44b70a55c73854e27e6da521dad9bb4') { throw 'Generated macros changed' }
uv run pytest
uv run ruff check .
git diff --check
```

Expected: all tests pass, Ruff reports success, and every guard exits 0.

- [ ] **Step 5: Commit the canonical manuscript checkpoint**

Run:

```powershell
git add paper/main.tex paper/main.pdf artifacts/run_provenance.json
git commit -m "polish arxiv manuscript"
```

Expected: one commit containing only the canonical manuscript, PDF, and coupled provenance update.

## Chunk 2: Upload archive and handoff

### Task 4: Build and independently verify the four-file archive

**Files:**
- Create: `paper/arxiv-source.zip`
- Source: `paper/main.tex`
- Source: `paper/references.bib`
- Source: `artifacts/analysis/paper_results.tex`
- Source: `artifacts/analysis/figures/risk_coverage_natural.pdf`

- [ ] **Step 1: Stage, build, archive, and clean up the source snapshot**

Run from the repository root:

```powershell
$stage = Join-Path ([IO.Path]::GetTempPath()) ('sash-arxiv-stage-' + [guid]::NewGuid())
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
try {
  New-Item -ItemType Directory $stage | Out-Null
  Copy-Item paper/main.tex,paper/references.bib,artifacts/analysis/paper_results.tex,artifacts/analysis/figures/risk_coverage_natural.pdf $stage
  $canonical = [IO.File]::ReadAllText((Resolve-Path paper/main.tex))
  $expectedMain = $canonical.Replace('../artifacts/analysis/paper_results.tex', 'paper_results.tex').Replace('../artifacts/analysis/figures/risk_coverage_natural.pdf', 'risk_coverage_natural.pdf')
  $stagedMain = Join-Path $stage 'main.tex'
  [IO.File]::WriteAllText($stagedMain, $expectedMain)
  $actualMain = [IO.File]::ReadAllText($stagedMain)
  if ($actualMain -cne $expectedMain) { throw 'Staged manuscript differs beyond the two path substitutions' }
  if ($actualMain -match '\.\./') { throw 'Parent path remains in staged main.tex' }
  $actualFiles = @(Get-ChildItem $stage -File | Select-Object -ExpandProperty Name | Sort-Object)
  $expectedFiles = @('main.tex','paper_results.tex','references.bib','risk_coverage_natural.pdf') | Sort-Object
  if (Compare-Object $expectedFiles $actualFiles) { throw "Unexpected staged files: $($actualFiles -join ', ')" }

  $tectonic = Join-Path $env:TEMP 'sash-tectonic-0.16.9\tectonic.exe'
  $env:SOURCE_DATE_EPOCH = '1783900800'
  Push-Location $stage
  try {
    & $tectonic main.tex --keep-logs --color never
    if ($LASTEXITCODE -ne 0) { throw 'Staged Tectonic build failed' }
    $bad = Select-String -Path main.log -Pattern 'undefined references|Citation .* undefined|LaTeX Error|Package .* Error' -CaseSensitive:$false
    if ($bad) { $bad; throw 'Staged log contains blocking diagnostics' }
  } finally {
    Pop-Location
  }

  $archive = Join-Path (Resolve-Path paper).Path 'arxiv-source.zip'
  Compress-Archive -Path @(
    (Join-Path $stage 'main.tex'),
    (Join-Path $stage 'references.bib'),
    (Join-Path $stage 'paper_results.tex'),
    (Join-Path $stage 'risk_coverage_natural.pdf')
  ) -DestinationPath $archive -CompressionLevel Optimal -Force
} finally {
  if (Test-Path $stage) {
    $stageFull = [IO.Path]::GetFullPath($stage)
    if (-not $stageFull.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cleanup path: $stageFull" }
    Remove-Item -LiteralPath $stageFull -Recurse -Force
  }
}
```

Expected: `paper/arxiv-source.zip` exists, the staged source built cleanly, and the temporary staging directory was removed.

- [ ] **Step 2: Extract and rebuild the final archive independently**

Run in a fresh PowerShell session:

```powershell
$verify = Join-Path ([IO.Path]::GetTempPath()) ('sash-arxiv-verify-' + [guid]::NewGuid())
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
try {
  New-Item -ItemType Directory $verify | Out-Null
  Expand-Archive paper/arxiv-source.zip $verify
  $actual = @(Get-ChildItem $verify -File | Select-Object -ExpandProperty Name | Sort-Object)
  $expected = @('main.tex','paper_results.tex','references.bib','risk_coverage_natural.pdf') | Sort-Object
  if (Compare-Object $expected $actual) { throw "Unexpected archive contents: $($actual -join ', ')" }
  $canonical = [IO.File]::ReadAllText((Resolve-Path paper/main.tex))
  $expectedMain = $canonical.Replace('../artifacts/analysis/paper_results.tex', 'paper_results.tex').Replace('../artifacts/analysis/figures/risk_coverage_natural.pdf', 'risk_coverage_natural.pdf')
  $archiveMain = [IO.File]::ReadAllText((Join-Path $verify 'main.tex'))
  if ($archiveMain -cne $expectedMain) { throw 'Archived manuscript differs beyond the two path substitutions' }
  if ($archiveMain -match '\.\./') { throw 'Parent path remains in archive' }

  $tectonic = Join-Path $env:TEMP 'sash-tectonic-0.16.9\tectonic.exe'
  $env:SOURCE_DATE_EPOCH = '1783900800'
  Push-Location $verify
  try {
    & $tectonic main.tex --keep-logs --color never
    if ($LASTEXITCODE -ne 0) { throw 'Archive Tectonic build failed' }
    $bad = Select-String -Path main.log -Pattern 'undefined references|Citation .* undefined|LaTeX Error|Package .* Error' -CaseSensitive:$false
    if ($bad) { $bad; throw 'Archive Tectonic log contains blocking diagnostics' }

    $pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
    $bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
    if ($pdflatex -and $bibtex) {
      Remove-Item main.aux,main.bbl,main.blg,main.log,main.out,main.pdf -Force -ErrorAction SilentlyContinue
      & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
      if ($LASTEXITCODE -ne 0) { throw 'PDFLaTeX pass 1 failed' }
      & $bibtex main
      if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed' }
      & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
      if ($LASTEXITCODE -ne 0) { throw 'PDFLaTeX pass 2 failed' }
      & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
      if ($LASTEXITCODE -ne 0) { throw 'PDFLaTeX pass 3 failed' }
      $bad = Select-String -Path main.log -Pattern 'undefined references|Citation .* undefined|LaTeX Error|Package .* Error' -CaseSensitive:$false
      if ($bad) { $bad; throw 'PDFLaTeX log contains blocking diagnostics' }
      'ARXIV_LOCAL_GATE=PDFLaTeX+BibTeX passed; AutoTeX preview still required'
    } else {
      'ARXIV_LOCAL_GATE=Tectonic-only; AutoTeX preview required'
    }
  } finally {
    Pop-Location
  }
} finally {
  if (Test-Path $verify) {
    $verifyFull = [IO.Path]::GetFullPath($verify)
    if (-not $verifyFull.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cleanup path: $verifyFull" }
    Remove-Item -LiteralPath $verifyFull -Recurse -Force
  }
}
```

Expected: exact file list, a clean Tectonic build, a conditional PDFLaTeX/BibTeX result, an explicit AutoTeX-preview boundary, and no leftover verification directory.

### Task 5: Document the handoff and run the final gate

**Files:**
- Modify: `README.md`
- Verify: `paper/arxiv-source.zip`

- [ ] **Step 1: Add concise arXiv instructions**

Use `apply_patch` to append:

```markdown
## arXiv upload

Upload `paper/arxiv-source.zip`. It contains only the manuscript, bibliography,
generated result macros, and the figure used by the paper. Inspect arXiv's generated
PDF preview before finalizing the submission; the local gate uses Tectonic 0.16.9,
not arXiv's AutoTeX environment.
```

- [ ] **Step 2: Run the complete final verification**

Use `@superpowers:verification-before-completion`, then run:

```powershell
$macroHash = (Get-FileHash artifacts/analysis/paper_results.tex -Algorithm SHA256).Hash.ToLowerInvariant()
if ($macroHash -ne 'f1b95c5c84e4f01d12a1adf5ec14dd1fa44b70a55c73854e27e6da521dad9bb4') { throw 'Generated macros changed' }
$p = Get-Content artifacts/run_provenance.json -Raw | ConvertFrom-Json
$pdfHash = (Get-FileHash paper/main.pdf -Algorithm SHA256).Hash.ToLowerInvariant()
if ($p.integrity.sha256.paper_pdf -ne $pdfHash) { throw 'Final PDF hash mismatch' }
if ($p.manuscript.pdf_bytes -ne (Get-Item paper/main.pdf).Length) { throw 'Final PDF byte mismatch' }
if (-not (Test-Path paper/arxiv-source.zip)) { throw 'arXiv archive missing' }
uv run pytest
uv run ruff check .
git diff --check
git status --short
```

Expected: tests and Ruff pass; only `README.md` and `paper/arxiv-source.zip` remain uncommitted.

- [ ] **Step 3: Commit the upload handoff**

Run:

```powershell
git add README.md paper/arxiv-source.zip
git commit -m "package arxiv source"
git status --short
```

Expected: clean worktree.

- [ ] **Step 4: Report the honest submission boundary**

Provide the clickable paths to `paper/main.pdf` and `paper/arxiv-source.zip`, the
test/build results, final PDF hash, and the unchanged macro hash. State explicitly
that arXiv's generated AutoTeX preview must still be inspected after upload.
