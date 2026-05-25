# ds-de-se-hub

Personal knowledge library for Data Science, Data Engineering, and Software Engineering — books, notebooks, reference PDFs, datasets, and code in one place.

---

## Purpose

- Single source-of-truth for all study material.
- Decade directory schema (`00_`, `10_`, …) — scales without renaming.
- Git LFS for large binaries so clones stay lightweight.
- Branch-protected `main` — every change goes through a PR.

---

## Folder Structure

```
.
├── 00_foundations/
│   ├── books/
│   ├── cv/
│   └── statistics/
├── 10_programming/
│   └── python/
│       ├── 00_books/
│       ├── 01_reference_pdfs/
│       ├── 02_notebooks/
│       ├── challenges/
│       ├── 90_playgrounds/
│       └── data/
├── 20_data_science/
│   ├── 21_ml_classical/
│   │   ├── 00_books/
│   │   ├── 01_references/
│   │   ├── 02_notebooks/
│   │   ├── challenges/
│   │   └── data/
│   ├── 22_deep_learning/
│   │   ├── 00_books/
│   │   └── 01_references/
│   ├── 23_specialisations/
│   │   ├── actuarial/
│   │   ├── nlp/
│   │   └── time_series/
│   └── data_school/
│       ├── ML-text-main/
│       └── ML-Workflow/
├── 30_data_engineering/
│   └── spark_big_data/
│       └── books/
├── 40_software_engineering/
│   ├── dsa_algorithms/
│   └── system_design/
├── 50_industry_and_research/
│   └── 00_ml_papers/
└── 60_interview_prep/
    ├── case_studies/
    └── coding_drills/
```

**Decade prefixes** leave slots (`70_`, `80_`) open for future tracks without renaming anything.

Each topic folder uses a consistent internal layout:

| Subfolder | What goes here |
|-----------|---------------|
| `00_books/` | Full textbooks and long-form PDFs |
| `01_references/` | Slide decks, cheat-sheets, papers, short guides |
| `02_notebooks/` | Jupyter notebooks and code-along tutorials |
| `challenges/` | Assignments, problem sets, self-challenges |
| `data/` | Sample datasets (CSVs, Excel, etc.) |
| `90_playgrounds/` | Exploratory / experimental code |

---

## Where to Put New Files

| Question | Answer |
|----------|--------|
| Which decade? | `00` foundations · `10` programming · `20` data science · `30` data engineering · `40` software engineering · `50` industry & research · `60` interview prep |
| Full textbook? | `<topic>/00_books/` |
| Short reference, paper, or cheat-sheet? | `<topic>/01_references/` |
| Notebook or tutorial? | `<topic>/02_notebooks/` |
| Assignment or problem set? | `<topic>/challenges/` |
| Dataset? | `<topic>/data/` — track via Git LFS |
| Large binary (PDF, image, dataset)? | Run `git lfs track "*.pdf"` before committing |
| Cross-topic resource? | File under primary learning intent (e.g. ML paper → `50_industry_and_research/`) |

---

## Commit & PR Workflow

```bash
# Start from an up-to-date main
git checkout main && git pull origin main

# Create a feature branch
git checkout -b <type>/<scope>/<description>
# e.g. feat/python/add-regex-notebooks

# Stage and commit
git add <files>
git commit -m "<type>(<scope>): <short description>"
# e.g. feat(ml): add clustering challenge notebooks

# Push and open a PR
git push -u origin <branch>
gh pr create --fill --web
```

Merge via GitHub UI (squash or rebase). Delete the branch after merge.

---

## Do's and Don'ts

**Do:**
- Use feature branches — no direct commits to `main`.
- Use snake_case filenames without spaces.
- Track large files with Git LFS before committing.
- Keep `main` up to date before branching (`git pull origin main`).

**Don't:**
- Commit `venv/`, `.venv/`, or any virtual environment folder.
- Commit sensitive data, API keys, or proprietary datasets.
- Commit large files without LFS — check with `git lfs status` if unsure.
- Rename decade folders — it breaks history and links.

---

## Maintainer

rv · `23rishavsharma@gmail.com` · open an issue for questions.

---

## Appendix: Issue-Resolution Log

| Issue | Fix |
|-------|-----|
| Infinite password loop on LFS push | Set `credential.helper` (`osxkeychain`), generate a PAT with `repo` + `write:packages` scopes, store with `git credential approve`. Load SSH key: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`. |
| `git pull` blocked by uncommitted renames | `git stash --include-untracked` → `git pull` → `git stash pop` → resolve conflicts → commit. |
| GitHub large-file warning / push failure >100MB | Adopt Git LFS. Migrate existing files: `git lfs migrate import --everything --include="*.pdf,*.png,*.zip"`. |
| macOS `Icon\r` artifacts in repo | Add `Icon\r` and `Icon?` to `.gitignore`. Remove existing: `find . -name "Icon?" -delete`. |
| LFS pointer conflict ("should have been pointers, but weren't") | `rm <file>` → `git lfs checkout <file>` → verify with `git lfs status`. |
| Should `.gitkeep` be tracked? | Keep in intentionally empty dirs. Remove once the folder has real content. |
