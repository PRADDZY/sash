# Minimal AI-Assistance Wording Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every Codex brand mention from the manuscript and retain only the approved generic disclosure in Section 7.

**Architecture:** Make three literal prose substitutions in `paper/main.tex`; no code, data, methods, or result changes. Rebuild the deterministic PDF and refresh only its byte count and SHA-256 provenance fields.

**Tech Stack:** LaTeX, Tectonic 0.16.9, PowerShell, pypdf

---

## Chunk 1: Manuscript and PDF

### Task 1: Apply and verify the approved disclosure wording

**Files:**
- Modify: `paper/main.tex:486`
- Modify: `paper/main.tex:572`
- Modify: `paper/main.tex:584-590`
- Modify: `paper/main.pdf`
- Modify: `artifacts/run_provenance.json:138-173`

- [ ] **Step 1: Confirm the old wording is present**

Run:

```powershell
$matches = Select-String -Path paper/main.tex -Pattern 'Codex' -CaseSensitive:$false
if ($matches.Count -ne 3) { throw "Expected 3 Codex mentions, found $($matches.Count)" }
```

Expected: exit 0 with exactly three matches.

- [ ] **Step 2: Make only the three approved prose substitutions**

In `paper/main.tex`, use `apply_patch` to produce exactly these passages:

```tex
We conduct a case-by-case review of the 15 highest-confidence
```

```tex
not an isolated treatment effect.  The single qualitative coding pass
```

```tex
Generative AI assisted with implementation, analysis, and drafting.  The author
verified all sources, artifacts, and claims and remains responsible for the work.
```

Delete the remainder of the old AI-assistance paragraph. Do not change the Section 7 title or any other prose.

- [ ] **Step 3: Prove that only the three approved source substitutions occurred**

Run:

```powershell
@'
from pathlib import Path
import subprocess

before = subprocess.check_output(
    ['git', 'show', 'HEAD:paper/main.tex'], text=True, encoding='utf-8'
)
after = Path('paper/main.tex').read_text(encoding='utf-8')
replacements = [
    (
        'We conduct a Codex-assisted case-by-case review',
        'We conduct a case-by-case review',
    ),
    (
        'The single Codex-assisted qualitative coding pass',
        'The single qualitative coding pass',
    ),
    (
        '''OpenAI Codex assisted with literature retrieval, protocol refinement, implementation,
experimental orchestration, statistical checks, qualitative failure categorization,
and manuscript drafting.  The
author selected the research question and checkpoints, directed the work, and is
responsible for source verification, interpretation, and every final numerical
claim.  No generated value is treated as an experimental result unless it is
reproduced from the archived prediction and analysis artifacts.''',
        '''Generative AI assisted with implementation, analysis, and drafting.  The author
verified all sources, artifacts, and claims and remains responsible for the work.''',
    ),
]
for old, new in replacements:
    assert before.count(old) == 1, old
    before = before.replace(old, new, 1)
assert after == before
assert 'codex' not in after.lower()
section = after.split(r'\section{Ethics and AI Assistance}', 1)[1].split(
    r'\section{Conclusion}', 1
)[0]
assert section.count('Generative AI assisted') == 1
print('SOURCE_DIFF_PASS')
'@ | uv run python -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: `SOURCE_DIFF_PASS`.

- [ ] **Step 4: Build the PDF once and retain an ignored comparison copy**

Run:

```powershell
$tectonic = 'C:\Users\Pratik Daithankar\AppData\Local\Temp\tectonic-0.16.9\tectonic.exe'
$env:SOURCE_DATE_EPOCH = '1783900800'
Push-Location paper
& $tectonic main.tex --keep-logs --color never
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
Copy-Item main.pdf '..\.modal\disclosure-first.pdf' -Force
Pop-Location
```

Expected: Tectonic exits 0 and `.modal/disclosure-first.pdf` exists. The comparison copy is ignored and is removed in Step 5.

- [ ] **Step 5: Build the PDF again and prove determinism**

```powershell
$tectonic = 'C:\Users\Pratik Daithankar\AppData\Local\Temp\tectonic-0.16.9\tectonic.exe'
$env:SOURCE_DATE_EPOCH = '1783900800'
$firstPdf = '.modal\disclosure-first.pdf'
if (-not (Test-Path $firstPdf)) { throw 'Missing first-build comparison PDF' }
Push-Location paper
& $tectonic main.tex --keep-logs --color never
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
Pop-Location
$firstPdfHash = (Get-FileHash $firstPdf -Algorithm SHA256).Hash.ToLowerInvariant()
$secondPdfHash = (Get-FileHash paper\main.pdf -Algorithm SHA256).Hash.ToLowerInvariant()
if ($firstPdfHash -ne $secondPdfHash) { throw "Nondeterministic PDF: $firstPdfHash != $secondPdfHash" }
Remove-Item -LiteralPath $firstPdf
```

Expected: Tectonic exits 0, the hashes are identical, and the ignored comparison copy is removed.

- [ ] **Step 6: Reject manuscript build warnings**

```powershell
$warnings = Select-String -Path paper/main.log -Pattern 'LaTeX Warning|Package .* Warning|Overfull|Underfull|undefined references|Citation .* undefined|There were undefined' -CaseSensitive:$false
if ($warnings) { $warnings | ForEach-Object { Write-Output $_.Line }; throw 'LaTeX log contains warnings' }
```

Expected: exit 0 with no matching log lines.

- [ ] **Step 7: Print the new PDF provenance values**

```powershell
$pdfHash = (Get-FileHash paper/main.pdf -Algorithm SHA256).Hash.ToLowerInvariant()
$pdfBytes = (Get-Item paper/main.pdf).Length
Write-Output "paper_pdf=$pdfHash"
Write-Output "pdf_bytes=$pdfBytes"
```

Expected: one lowercase SHA-256 and one positive byte count.

- [ ] **Step 8: Refresh only the two PDF provenance fields**

Use `apply_patch` to replace `integrity.sha256.paper_pdf` with the literal `paper_pdf` value and `manuscript.pdf_bytes` with the literal `pdf_bytes` value printed in Step 7. Do not change any other provenance field.

- [ ] **Step 9: Verify the rendered disclosure and unchanged headline results**

Run:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
@'
import re
from pypdf import PdfReader

reader = PdfReader('paper/main.pdf')
text = re.sub(r'\s+', ' ', ' '.join(page.extract_text() or '' for page in reader.pages))
disclosure = (
    'Generative AI assisted with implementation, analysis, and drafting. '
    'The author verified all sources, artifacts, and claims and remains responsible for the work.'
)
assert len(reader.pages) == 8
assert 'codex' not in text.lower()
assert text.count(disclosure) == 1
section_start = text.index('7 Ethics and AI Assistance')
section_end = text.index('8 Conclusion')
assert disclosure in text[section_start:section_end]
assert 'Generative AI assisted' not in text[:section_start] + text[section_end:]
for value in ['69.5%', '72.6%', '20.5%', '16.6%', '15.71%', '41.28%']:
    assert value in text, value
print('PDF_DISCLOSURE_PASS pages=8')
'@ | uv run --with pypdf python -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: `PDF_DISCLOSURE_PASS pages=8`. `pypdf` is supplied ephemerally by `uv --with`; it is not added to project dependencies.

- [ ] **Step 10: Run the existing test suite**

```powershell
uv run pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: 28 tests pass.

- [ ] **Step 11: Run lint**

```powershell
uv run ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected: `All checks passed!`.

- [ ] **Step 12: Validate every recorded provenance hash**

```powershell
$p = Get-Content artifacts/run_provenance.json -Raw | ConvertFrom-Json
$paths = [ordered]@{
  natural_and_shift_manifest = 'artifacts/manifest.jsonl'
  pilot_manifest = '.modal/pilot_manifest.jsonl'
  base_predictions = 'artifacts/predictions/base.jsonl'
  public_tuned_predictions = 'artifacts/predictions/finetuned.jsonl'
  metrics_csv = 'artifacts/analysis/metrics.csv'
  paired_differences_csv = 'artifacts/analysis/paired_differences.csv'
  subgroup_metrics_csv = 'artifacts/analysis/subgroup_metrics.csv'
  failure_review_csv = 'artifacts/analysis/failure_review.csv'
  publication_examples_csv = 'artifacts/analysis/publication_examples.csv'
  paper_results_tex = 'artifacts/analysis/paper_results.tex'
  paper_pdf = 'paper/main.pdf'
  final_modal_analysis_bundle = 'artifacts/analysis_bundle.zip'
}
foreach ($item in $paths.GetEnumerator()) {
  $actual = (Get-FileHash $item.Value -Algorithm SHA256).Hash.ToLowerInvariant()
  $expected = [string]$p.integrity.sha256.($item.Key)
  if ($actual -ne $expected) { throw "$($item.Key): $actual != $expected" }
}
if ((Get-Item paper/main.pdf).Length -ne $p.manuscript.pdf_bytes) { throw 'PDF byte count mismatch' }
Write-Output 'PROVENANCE_PASS hashes=12'
```

Expected: `PROVENANCE_PASS hashes=12`.

- [ ] **Step 13: Check the final diff contains no unrelated tracked changes**

```powershell
git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git status --short
git diff -- paper/main.tex artifacts/run_provenance.json
```

Expected: the plan document plus `paper/main.tex`, `paper/main.pdf`, and `artifacts/run_provenance.json` are the only uncommitted files; the manuscript diff is exactly the three approved substitutions and provenance changes only the PDF hash and byte count.

- [ ] **Step 14: Commit the implementation plan**

```powershell
git add docs/superpowers/plans/2026-07-13-minimal-ai-assistance-wording.md
git commit -m "plan disclosure edit"
```

- [ ] **Step 15: Commit the coupled manuscript artifact update**

```powershell
git add paper/main.tex paper/main.pdf artifacts/run_provenance.json
git commit -m "simplify AI disclosure"
```
